"""
The step plan: which records are in step N, decided before training starts.

The plan depends only on the seed, the stage settings and the data, never on
how far training has got. Resuming at step N therefore continues with exactly
the steps an uninterrupted run would have taken.

Rules:
  - a step holds records of one source only (mixing sources lets the model
    tell passages apart by source style, which is a shortcut);
  - dev records never appear;
  - each source is shuffled once per epoch and cut into groups of
    `batch_queries`; the last partial group of an epoch is dropped;
  - the groups of all sources and epochs are then shuffled together.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .cache import SourceCache, TokenCache
from .config import WEBFAQ_SOURCES, StageConfig
from .errors import DataError


@dataclass
class StepPlan:
    stage: int
    sources: list[str]
    source_of_step: np.ndarray  # int16 [steps], index into `sources`
    epoch_of_step: np.ndarray  # int16 [steps]
    records: np.ndarray  # int32 [steps, batch_queries], record indices in the source

    @property
    def steps(self) -> int:
        return len(self.source_of_step)

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.stage}|{'|'.join(self.sources)}".encode("utf-8"))
        for array in (self.source_of_step, self.epoch_of_step, self.records):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()[:16]

    def step(self, index: int) -> tuple[str, int, np.ndarray]:
        return (
            self.sources[int(self.source_of_step[index])],
            int(self.epoch_of_step[index]),
            self.records[index].astype(np.int64),
        )


def stage_records(source: SourceCache, stage: StageConfig, cache: TokenCache, seed: int) -> np.ndarray:
    """
    The training records of `source` a stage uses, sorted.

    Everything except dev, unless the stage takes only a sample of WebFAQ: then
    `webfaq_sample` records are split over the WebFAQ sources in proportion to
    their size and drawn once, by seed. `source_records` caps any source the
    same way. Mining uses the same function, so it mines exactly the records
    stage 2 will train on.
    """
    records = _webfaq_records(source, stage, cache, seed)
    cap = stage.source_records.get(source.name, 0)
    if cap and cap < len(records):
        rng = np.random.default_rng([seed, 11, int(hashlib.sha256(source.name.encode()).hexdigest()[:8], 16)])
        records = np.sort(rng.choice(records, size=cap, replace=False))
    return records


def _webfaq_records(source: SourceCache, stage: StageConfig, cache: TokenCache, seed: int) -> np.ndarray:
    records = source.train_records
    if not source.is_webfaq or stage.webfaq_sample <= 0:
        return records
    webfaq = [name for name in cache.names if name in WEBFAQ_SOURCES]
    total = sum(len(cache[name].train_records) for name in webfaq)
    if stage.webfaq_sample >= total:
        return records
    share = int(round(stage.webfaq_sample * len(records) / total))
    rng = np.random.default_rng([seed, 7, webfaq.index(source.name)])
    return np.sort(rng.choice(records, size=share, replace=False))


def build_plan(cache: TokenCache, stage: StageConfig, stage_number: int, seed: int) -> StepPlan:
    names = cache.names
    groups: list[np.ndarray] = []
    group_source: list[int] = []
    group_epoch: list[int] = []
    for source_index, name in enumerate(names):
        source = cache[name]
        records = stage_records(source, stage, cache, seed)
        epochs = stage.source_epochs.get(name, stage.epochs_webfaq if source.is_webfaq else stage.epochs_curated)
        full = len(records) // stage.batch_queries
        if full == 0:
            continue  # smaller than one step: the source cannot fill a batch
        for epoch in range(epochs):
            rng = np.random.default_rng([seed, stage_number, source_index, epoch])
            order = rng.permutation(records)[: full * stage.batch_queries]
            groups.append(order.reshape(full, stage.batch_queries).astype(np.int32))
            group_source.extend([source_index] * full)
            group_epoch.extend([epoch] * full)
    if not groups:
        raise DataError("no source has enough records to fill one step")
    records = np.concatenate(groups)
    order = np.random.default_rng([seed, stage_number, 99]).permutation(len(records))
    return StepPlan(
        stage=stage_number,
        sources=names,
        source_of_step=np.asarray(group_source, dtype=np.int16)[order],
        epoch_of_step=np.asarray(group_epoch, dtype=np.int16)[order],
        records=records[order],
    )


def positive_choice(records: np.ndarray, positives: np.ndarray, epoch: int, seed: int) -> np.ndarray:
    """
    Which positive of each record a visit uses.

    A record with several positives starts at a seed-dependent one and moves to
    the next on every epoch, so over the epochs each positive gets its turn.
    """
    return (records + seed + epoch) % positives
