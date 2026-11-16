"""
From a step of the plan to tensors.

A training step is a list of texts: the queries first, then the candidates
(one positive per query, then the hard negatives). All of them go through the
same encoder, so they are sorted by length together and cut into chunks of at
most `chunk_tokens` padded tokens; that keeps padding to a few percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .cache import SourceCache
from .plan import positive_choice


@dataclass
class StepBatch:
    """The texts of one step and what the loss needs to know about them."""

    source: str
    tokens: list[np.ndarray]  # queries, then candidates
    queries: int
    target: np.ndarray  # int64 [queries], index of each query's positive among the candidates
    mask: np.ndarray  # bool [queries, candidates], True = not a valid negative for this query

    @property
    def candidates(self) -> int:
        return len(self.tokens) - self.queries


@dataclass
class Chunk:
    index: torch.Tensor  # int64 [n], positions of these texts in the step
    input_ids: torch.Tensor  # int64 [n, length]
    attention_mask: torch.Tensor  # int64 [n, length]

    @property
    def padded_tokens(self) -> int:
        return self.input_ids.numel()

    def to(self, device: torch.device) -> "Chunk":
        return Chunk(
            self.index.to(device, non_blocking=True),
            self.input_ids.to(device, non_blocking=True),
            self.attention_mask.to(device, non_blocking=True),
        )

    def pin(self) -> "Chunk":
        return Chunk(self.index.pin_memory(), self.input_ids.pin_memory(), self.attention_mask.pin_memory())


def step_passages(
    source: SourceCache,
    records: np.ndarray,
    epoch: int,
    seed: int,
    hard_negatives: int,
    mined: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The passages of one step: (one positive per record, the hard negatives).

    `mined` is the int32 [records of the source, k] table written by the miner
    (passage indices, -1 = none). A record's own negatives are used first, then
    mined ones, up to `hard_negatives`; a record with fewer simply contributes
    fewer candidates.
    """
    starts = source.record_start[records]
    counts = source.record_positives[records]
    positives = starts + positive_choice(records, counts, epoch, seed)

    negatives: list[int] = []
    if hard_negatives > 0:
        ends = source.record_start[records + 1]
        for position, record in enumerate(records):
            own = np.arange(starts[position] + counts[position], ends[position])
            if len(own) > hard_negatives:
                # Rotate through a record's own negatives across epochs.
                shift = (int(record) + seed + epoch) % len(own)
                own = np.roll(own, -shift)[:hard_negatives]
            chosen = list(own)
            if mined is not None and len(chosen) < hard_negatives:
                extra = mined[record]
                chosen.extend(int(p) for p in extra[extra >= 0][: hard_negatives - len(chosen)])
            negatives.extend(chosen)
    return positives, np.asarray(negatives, dtype=np.int64)


def build_step(
    source: SourceCache,
    records: np.ndarray,
    epoch: int,
    seed: int,
    hard_negatives: int,
    mined: np.ndarray | None = None,
) -> StepBatch:
    """Assemble one step from the cache: tokens, targets and the false-negative mask."""
    starts = source.record_start[records]
    counts = source.record_positives[records]
    positives, negatives = step_passages(source, records, epoch, seed, hard_negatives, mined)
    candidate_passages = np.concatenate([positives, negatives])

    # A candidate is not a negative for query i when its text equals any of
    # query i's positives -- the one it is paired with excepted, that is its target.
    candidate_hash = source.passage_hash[candidate_passages]
    width = int(counts.max())
    own_hash = np.zeros((len(records), width), dtype=np.uint64)
    valid = np.zeros((len(records), width), dtype=bool)
    for column in range(width):
        has = counts > column
        own_hash[has, column] = source.passage_hash[starts[has] + column]
        valid[has, column] = True
    mask = np.zeros((len(records), len(candidate_passages)), dtype=bool)
    for column in range(width):
        mask |= (own_hash[:, column, None] == candidate_hash[None, :]) & valid[:, column, None]
    target = np.arange(len(records), dtype=np.int64)
    mask[target, target] = False

    tokens = [source.query(int(record)) for record in records]
    tokens.extend(source.passage(int(passage)) for passage in candidate_passages)
    return StepBatch(source.name, tokens, len(records), target, mask)


def chunk_order(lengths: np.ndarray, budget: int) -> list[np.ndarray]:
    """
    Group text positions into chunks of at most `budget` padded tokens.

    Longest first: the first chunk is then the most expensive one, so a budget
    that is too large for the device fails on the first chunk, not hours in.
    """
    order = np.argsort(-lengths, kind="stable")
    chunks: list[np.ndarray] = []
    start = 0
    while start < len(order):
        longest = int(lengths[order[start]])
        size = max(1, budget // max(longest, 1))
        chunks.append(order[start : start + size])
        start += size
    return chunks


def padded_tokens(lengths: np.ndarray, budget: int) -> int:
    """Padded tokens `make_chunks` would produce for texts of these lengths."""
    return int(sum(int(lengths[group[0]]) * len(group) for group in chunk_order(lengths, budget)))


def make_chunks(tokens: list[np.ndarray], budget: int, pad_id: int = 0) -> list[Chunk]:
    lengths = np.fromiter((len(ids) for ids in tokens), dtype=np.int64, count=len(tokens))
    chunks = []
    for group in chunk_order(lengths, budget):
        width = int(lengths[group[0]])
        ids = np.full((len(group), width), pad_id, dtype=np.int64)
        mask = np.zeros((len(group), width), dtype=np.int64)
        for row, position in enumerate(group):
            length = lengths[position]
            ids[row, :length] = tokens[position]
            mask[row, :length] = 1
        chunks.append(
            Chunk(torch.from_numpy(group.astype(np.int64)), torch.from_numpy(ids), torch.from_numpy(mask))
        )
    return chunks
