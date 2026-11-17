"""
Hard-negative mining for stage 2.

With the stage-1 adapter, for every record stage 2 will train on:

  1. rank the passages of the same source by cosine to the query (top `top_k`);
  2. drop the record's own passages (by text), the first `skip_top` ranks
     (too likely to be unlabelled positives) and anything scoring above
     `max_score_ratio` x the score of the record's best positive;
  3. keep `per_query` of what is left, drawn at random by seed.

The result is one table per source, `mined/<source>.npy`: int32
[records of the source, per_query] of passage indices into the token cache,
-1 where nothing was kept. `mined.json` ties the tables to the cache they index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .cache import SourceCache, TokenCache
from .config import EmbeddingConfig
from .errors import DataError
from .encoding import encode_tokens
from .plan import stage_records
from .resources import DevicePlan

MINED_FILE = "mined.json"


def mine_source(
    encoder: nn.Module,
    source: SourceCache,
    records: np.ndarray,
    config: EmbeddingConfig,
    plan: DevicePlan,
    seed: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    mining = config.mining
    table = np.full((source.records, mining.per_query), -1, dtype=np.int32)

    # The pool: every distinct passage text of the records being mined.
    spans = [np.arange(source.record_start[r], source.record_start[r + 1]) for r in records]
    passages = np.concatenate(spans)
    _, first = np.unique(source.passage_hash[passages], return_index=True)
    pool = passages[np.sort(first)]
    pool_vectors = encode_tokens(encoder, [source.passage(int(p)) for p in pool], plan)
    pool_hash = torch.from_numpy(source.passage_hash[pool].view(np.int64)).to(plan.device)
    pool_index = torch.from_numpy(pool).to(plan.device)
    query_vectors = encode_tokens(encoder, [source.query(int(r)) for r in records], plan)

    # Each record's own passage hashes, padded, to recognise them among the results.
    width = max(len(span) for span in spans)
    own = np.zeros((len(records), width), dtype=np.int64)
    own_valid = np.zeros((len(records), width), dtype=bool)
    positive = np.zeros((len(records), width), dtype=bool)
    for row, span in enumerate(spans):
        own[row, : len(span)] = source.passage_hash[span].view(np.int64)
        own_valid[row, : len(span)] = True
        positive[row, : source.record_positives[records[row]]] = True
    own_t = torch.from_numpy(own).to(plan.device)
    own_valid_t = torch.from_numpy(own_valid).to(plan.device)
    positive_t = torch.from_numpy(positive).to(plan.device)
    hash_to_pool = {int(h): i for i, h in enumerate(pool_hash.tolist())}
    own_pool = torch.from_numpy(
        np.asarray([[hash_to_pool[int(h)] if v else 0 for h, v in zip(hr, vr)] for hr, vr in zip(own, own_valid)])
    ).to(plan.device)

    generator = torch.Generator(device="cpu").manual_seed(int(np.random.default_rng(seed).integers(2**62)))
    depth = min(mining.top_k, len(pool))
    kept = 0
    for start in range(0, len(records), 512):
        block = slice(start, start + 512)
        queries = query_vectors[block]
        similarity = queries @ pool_vectors.T
        scores, top = torch.topk(similarity, depth, dim=1)
        own_scores = torch.gather(similarity, 1, own_pool[block])
        best_positive = own_scores.masked_fill(~positive_t[block], -2.0).max(dim=1).values
        is_own = (
            (pool_hash[top].unsqueeze(2) == own_t[block].unsqueeze(1)) & own_valid_t[block].unsqueeze(1)
        ).any(dim=2)
        rank = torch.arange(depth, device=plan.device).unsqueeze(0)
        valid = (
            ~is_own
            & (rank >= mining.skip_top)
            & (scores <= mining.max_score_ratio * best_positive.unsqueeze(1))
        )
        draw = torch.rand(valid.shape, generator=generator).to(plan.device)
        draw = draw.masked_fill(~valid, -1.0)
        picked_draw, picked = torch.topk(draw, min(mining.per_query, depth), dim=1)
        chosen = pool_index[torch.gather(top, 1, picked)]
        chosen = chosen.masked_fill(picked_draw < 0, -1).cpu().numpy().astype(np.int32)
        table[records[block], : chosen.shape[1]] = chosen
        kept += int((chosen >= 0).sum())
    stats = {
        "records": int(len(records)),
        "pool": int(len(pool)),
        "negatives": kept,
        "per_record": kept / max(1, len(records)),
    }
    return table, stats


def mine(
    encoder: nn.Module,
    cache: TokenCache,
    config: EmbeddingConfig,
    plan: DevicePlan,
    output: str | Path,
    adapter_info: dict[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Mine every source for stage 2 and write the tables to `output`."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed = config.resources.seed
    document: dict[str, Any] = {
        "cache_digest": cache.digest,
        "mining": config.to_dict()["mining"],
        "adapter": adapter_info,
        "sources": {},
    }
    for index, name in enumerate(cache.names):
        source = cache[name]
        records = stage_records(source, config.stage_2, cache, seed)
        table, stats = mine_source(encoder, source, records, config, plan, [seed, 13, index])
        np.save(output / f"{name}.npy", table)
        document["sources"][name] = stats
        log(f"  {name}: {stats['negatives']:,} negatives for {stats['records']:,} records "
            f"(pool {stats['pool']:,})")
    with (output / MINED_FILE).open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return document


def load_mined(directory: str | Path, cache: TokenCache) -> dict[str, np.ndarray]:
    """The mined tables, after checking they index this token cache."""
    directory = Path(directory)
    path = directory / MINED_FILE
    if not path.is_file():
        raise DataError(f"no {MINED_FILE} in {directory}; run `embed.py mine` first")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document["cache_digest"] != cache.digest:
        raise DataError(
            f"the mined negatives in {directory} index a different token cache; mine again"
        )
    tables = {}
    for name in document["sources"]:
        table = np.load(directory / f"{name}.npy")
        if table.shape[0] != cache[name].records:
            raise DataError(f"{directory}/{name}.npy does not match the cache")
        tables[name] = table
    return tables
