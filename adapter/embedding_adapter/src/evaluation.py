"""
Measuring an adapter.

Two measurements, both nDCG@10:

  - **dev**: the held-out queries of each training source against their own
    positives plus a fixed sample of that source's training passages. Used to
    pick the best checkpoint. Never the benchmark.
  - **benchmark**: the 10 retrieval tasks and STS-B under `eval_root/tasks`,
    scored exactly like the reference script that produced the baseline
    numbers (`eval_root/reference_scripts/score.py`), so the adapter's numbers
    can be put next to them.

Benchmark protocol: 512 tokens; `query: ` for queries and `passage: ` for
documents, except arguana and quora, whose queries are documents themselves
(both sides `passage: `, and a document never retrieves itself); STS encodes
both sentences as queries and reports Spearman of the cosine.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .cache import SourceCache, TokenCache
from .config import EmbeddingConfig
from .encoding import encode_tokens
from .errors import DataError
from .pooling import normalize
from .resources import DevicePlan
from .text import TextTokenizer

RETRIEVAL_TASKS = (
    "scifact", "nfcorpus", "arguana", "fiqa", "scidocs",
    "trec-covid", "quora", "nq", "hotpotqa", "msmarco",
)  # fmt: skip
SYMMETRIC_TASKS = frozenset({"arguana", "quora"})
STS_TASK = "sts"
ALL_TASKS = RETRIEVAL_TASKS + (STS_TASK,)


def ndcg_at_10(
    queries: torch.Tensor,
    documents: torch.Tensor,
    gains: Sequence[dict[int, float]],
    exclude: Sequence[int | None] | None = None,
    dim: int | None = None,
) -> tuple[float, float]:
    """
    Mean nDCG@10 and recall@100 of unit `queries` against unit `documents`.

    `gains[i]` maps document index to relevance for query i. `exclude[i]` is a
    document index query i must not retrieve (itself, in symmetric tasks).
    """
    if dim is not None:
        queries, documents = normalize(queries, dim), normalize(documents, dim)
    depth = min(100, documents.shape[0])
    discounts = 1.0 / np.log2(np.arange(10) + 2)
    ndcg, recall = [], []
    for start in range(0, queries.shape[0], 256):
        scores = queries[start : start + 256] @ documents.T
        if exclude is not None:
            for row, document in enumerate(exclude[start : start + 256]):
                if document is not None:
                    scores[row, document] = -1e9
        top = torch.topk(scores, depth, dim=1).indices.cpu().numpy()
        for row in range(top.shape[0]):
            relevant = gains[start + row]
            ranked = top[row]
            found = np.asarray([relevant.get(int(d), 0.0) for d in ranked[:10]])
            dcg = float((found * discounts[: len(found)]).sum())
            ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:10]
            idcg = float((np.asarray(ideal) * discounts[: len(ideal)]).sum())
            ndcg.append(dcg / idcg if idcg > 0 else 0.0)
            positives = {d for d, g in relevant.items() if g > 0}
            hits = len(positives.intersection(int(d) for d in ranked))
            recall.append(hits / len(positives) if positives else 0.0)
    return float(np.mean(ndcg)), float(np.mean(recall))


# --------------------------------------------------------------------------
# dev
# --------------------------------------------------------------------------


@dataclass
class DevTask:
    name: str
    queries: list[np.ndarray]
    documents: list[np.ndarray]
    gains: list[dict[int, float]]


def build_dev_tasks(cache: TokenCache, config: EmbeddingConfig) -> list[DevTask]:
    """One retrieval task per source that has dev queries; fixed by the seed."""
    tasks = []
    for index, name in enumerate(cache.names):
        source = cache[name]
        if len(source.dev_records) == 0:
            continue
        tasks.append(_dev_task(source, config.dev.distractors_per_source, [config.resources.seed, 11, index]))
    return tasks


def _dev_task(source: SourceCache, distractors: int, seed: list[int]) -> DevTask:
    dev = source.dev_records
    own = np.concatenate(
        [np.arange(source.record_start[r], source.record_start[r + 1]) for r in dev]
    )
    # Distractors: passages of training records, so dev passages are not counted twice.
    train = source.train_records
    rng = np.random.default_rng(seed)
    picked = rng.choice(train, size=min(distractors, len(train)), replace=False)
    extra = source.record_start[picked] + (picked % source.record_positives[picked])
    passages = np.concatenate([own, extra])

    # One document per distinct text; a query's relevant documents are found by text.
    hashes = source.passage_hash[passages]
    _, first = np.unique(hashes, return_index=True)
    first.sort()
    passages, hashes = passages[first], hashes[first]
    by_hash = {int(h): i for i, h in enumerate(hashes)}

    gains = []
    for record in dev:
        start = source.record_start[record]
        positives = source.passage_hash[start : start + source.record_positives[record]]
        gains.append({by_hash[int(h)]: 1.0 for h in positives})
    return DevTask(
        name=source.name,
        queries=[source.query(int(r)) for r in dev],
        documents=[source.passage(int(p)) for p in passages],
        gains=gains,
    )


def evaluate_dev(encoder: nn.Module, tasks: Sequence[DevTask], plan: DevicePlan) -> dict[str, float]:
    """nDCG@10 per source and their plain mean (`dev_ndcg10`)."""
    result: dict[str, float] = {}
    for task in tasks:
        vectors = encode_tokens(encoder, task.queries + task.documents, plan)
        queries, documents = vectors[: len(task.queries)], vectors[len(task.queries) :]
        result[f"dev/{task.name}"] = ndcg_at_10(queries, documents, task.gains)[0]
    result["dev_ndcg10"] = float(np.mean([result[f"dev/{task.name}"] for task in tasks]))
    return result


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


@dataclass
class Benchmark:
    """The benchmark tasks, tokenised once and kept for repeated evaluations."""

    config: EmbeddingConfig
    tokenizer: TextTokenizer
    _loaded: dict[str, dict] = field(default_factory=dict)

    def _task(self, name: str) -> dict:
        if name in self._loaded:
            return self._loaded[name]
        path = Path(self.config.eval_root) / "tasks" / f"{name}.pkl"
        if not path.is_file():
            raise DataError(f"benchmark task not found: {path}")
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        text = self.config.text
        limit = text.eval_max_tokens

        def tokens(texts: list[str], prefix: str) -> list[np.ndarray]:
            output: list[np.ndarray] = []
            for start in range(0, len(texts), 8192):
                output.extend(self.tokenizer.encode_batch(texts[start : start + 8192], prefix, limit))
            return output

        if name == STS_TASK:
            task = {
                "a": tokens(raw["a"], text.query_prefix),
                "b": tokens(raw["b"], text.query_prefix),
                "gold": np.asarray(raw["gold"], dtype=np.float64),
            }
        else:
            symmetric = name in SYMMETRIC_TASKS
            index = {document: i for i, document in enumerate(raw["dids"])}
            # A judged document that is not in the corpus still counts in the ideal
            # ranking, as in the reference script; it gets an index nothing retrieves.
            absent: dict[str, int] = {}
            gains, exclude = [], []
            for query in raw["qids"]:
                relevant = raw["rel"][query]
                gain: dict[int, float] = {}
                for document, grade in relevant.items():
                    if symmetric and document == query:
                        continue
                    if document not in index and document not in absent:
                        absent[document] = -1 - len(absent)
                    gain[index[document] if document in index else absent[document]] = float(grade)
                gains.append(gain)
                exclude.append(index.get(query) if symmetric else None)
            task = {
                "q": tokens(raw["q"], text.passage_prefix if symmetric else text.query_prefix),
                "d": tokens(raw["d"], text.passage_prefix),
                "gains": gains,
                "exclude": exclude if symmetric else None,
            }
        self._loaded[name] = task
        return task

    def tokens(self, name: str) -> int:
        """Real (unpadded) tokens one evaluation of `name` encodes."""
        task = self._task(name)
        keys = ("a", "b") if name == STS_TASK else ("q", "d")
        return int(sum(len(ids) for key in keys for ids in task[key]))

    def lengths(self, name: str) -> np.ndarray:
        task = self._task(name)
        keys = ("a", "b") if name == STS_TASK else ("q", "d")
        return np.asarray([len(ids) for key in keys for ids in task[key]], dtype=np.int64)

    def evaluate(
        self,
        encoder: nn.Module,
        names: Sequence[str],
        plan: DevicePlan,
        dims: Sequence[int | None] = (None,),
    ) -> dict[str, float]:
        """
        `bench/<task>` (nDCG@10, or Spearman for STS) at full dimension;
        `bench/<task>@<d>` for every other entry of `dims`.
        """
        from scipy.stats import spearmanr

        result: dict[str, float] = {}
        for name in names:
            task = self._task(name)
            if name == STS_TASK:
                vectors = encode_tokens(encoder, task["a"] + task["b"], plan)
                a, b = vectors[: len(task["a"])], vectors[len(task["a"]) :]
                for dim in dims:
                    cosine = (normalize(a, dim) * normalize(b, dim)).sum(dim=1).cpu().numpy()
                    result[_key(name, dim)] = float(spearmanr(cosine, task["gold"]).correlation)
            else:
                vectors = encode_tokens(encoder, task["q"] + task["d"], plan)
                queries, documents = vectors[: len(task["q"])], vectors[len(task["q"]) :]
                for dim in dims:
                    ndcg, recall = ndcg_at_10(queries, documents, task["gains"], task["exclude"], dim)
                    result[_key(name, dim)] = ndcg
                    result[_key(name, dim).replace("bench/", "recall100/")] = recall
                del vectors, queries, documents
        return result


def _key(name: str, dim: int | None) -> str:
    return f"bench/{name}" if dim is None else f"bench/{name}@{dim}"


def retrieval_average(result: dict[str, float], dim: int | None = None) -> float | None:
    """Mean nDCG@10 over the 10 retrieval tasks, or None when some are missing."""
    keys = [_key(name, dim) for name in RETRIEVAL_TASKS]
    if not all(key in result for key in keys):
        return None
    return float(np.mean([result[key] for key in keys]))
