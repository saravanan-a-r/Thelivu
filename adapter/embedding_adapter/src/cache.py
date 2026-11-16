"""
The token cache: every training text tokenised once, stored as flat arrays.

`embed.py prepare` reads `data_root/<source>/train.jsonl` and writes, per
source, under `cache_root/<source>/`:

    query_tokens.npy      uint16   all queries, concatenated
    query_offsets.npy     int64    [records + 1]
    passage_tokens.npy    uint16   all passages (positives, then negatives, per record)
    passage_offsets.npy   int64    [passages + 1]
    passage_hash.npy      uint64   text hash of each passage (false-negative mask)
    record_start.npy      int64    [records + 1], first passage of each record
    record_positives.npy  uint16   number of positives of each record
    split_key.npy         uint64   dev-split key of each record

Record `r` is line `r` of train.jsonl. Its passages are
`record_start[r] .. record_start[r + 1]`; the first `record_positives[r]` of
them are positives, the rest are the record's own hard negatives.

Training never touches text again, so a step costs no tokenisation and the
exact token count of every stage is known before any GPU time is spent.
`cache.json` records what the cache was built from; opening a cache whose
inputs have changed is an error, not a silent mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from .config import WEBFAQ_SOURCES, EmbeddingConfig
from .errors import DataError
from .text import TOKEN_DTYPE, TextTokenizer, format_passage, split_key, text_hash

CACHE_FILE = "cache.json"
TRAIN_FILE = "train.jsonl"
CACHE_VERSION = 1
SHARD_BYTES = 48 * 1024 * 1024
ARRAYS = {
    "query_tokens": TOKEN_DTYPE,
    "query_offsets": np.int64,
    "passage_tokens": TOKEN_DTYPE,
    "passage_offsets": np.int64,
    "passage_hash": np.uint64,
    "record_start": np.int64,
    "record_positives": np.uint16,
    "split_key": np.uint64,
}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


@dataclass
class SourceCache:
    """The arrays of one source, memory-mapped."""

    name: str
    directory: Path
    dev_queries: int

    def _array(self, name: str) -> np.ndarray:
        return np.load(self.directory / f"{name}.npy", mmap_mode="r")

    @cached_property
    def query_tokens(self) -> np.ndarray:
        return self._array("query_tokens")

    @cached_property
    def query_offsets(self) -> np.ndarray:
        return np.asarray(self._array("query_offsets"))

    @cached_property
    def passage_tokens(self) -> np.ndarray:
        return self._array("passage_tokens")

    @cached_property
    def passage_offsets(self) -> np.ndarray:
        return np.asarray(self._array("passage_offsets"))

    @cached_property
    def passage_hash(self) -> np.ndarray:
        return np.asarray(self._array("passage_hash"))

    @cached_property
    def record_start(self) -> np.ndarray:
        return np.asarray(self._array("record_start"))

    @cached_property
    def record_positives(self) -> np.ndarray:
        return np.asarray(self._array("record_positives")).astype(np.int64)

    @property
    def records(self) -> int:
        return len(self.query_offsets) - 1

    @property
    def passages(self) -> int:
        return len(self.passage_offsets) - 1

    @property
    def is_webfaq(self) -> bool:
        return self.name in WEBFAQ_SOURCES

    @cached_property
    def dev_records(self) -> np.ndarray:
        """Record indices held out of training: the lowest split keys, sorted by index."""
        if self.dev_queries <= 0:
            return np.empty(0, dtype=np.int64)
        keys = np.asarray(self._array("split_key"))
        count = min(self.dev_queries, len(keys))
        order = np.argsort(keys, kind="stable")[:count]
        return np.sort(order).astype(np.int64)

    @cached_property
    def train_records(self) -> np.ndarray:
        mask = np.ones(self.records, dtype=bool)
        mask[self.dev_records] = False
        return np.flatnonzero(mask).astype(np.int64)

    @cached_property
    def query_lengths(self) -> np.ndarray:
        return np.diff(self.query_offsets)

    @cached_property
    def passage_lengths(self) -> np.ndarray:
        return np.diff(self.passage_offsets)

    def query(self, record: int) -> np.ndarray:
        return self.query_tokens[self.query_offsets[record] : self.query_offsets[record + 1]]

    def passage(self, passage: int) -> np.ndarray:
        return self.passage_tokens[self.passage_offsets[passage] : self.passage_offsets[passage + 1]]


@dataclass
class TokenCache:
    root: Path
    document: dict[str, Any]
    sources: dict[str, SourceCache]

    @property
    def names(self) -> list[str]:
        return list(self.sources)

    def __getitem__(self, name: str) -> SourceCache:
        return self.sources[name]

    @property
    def digest(self) -> str:
        return self.document["digest"]


def open_cache(config: EmbeddingConfig, tokenizer: TextTokenizer) -> TokenCache:
    """Open the cache and check it was built from today's data and settings."""
    path = config.cache_root / CACHE_FILE
    if not path.is_file():
        raise DataError(
            f"no token cache at {config.cache_root}; run `embed.py prepare` first"
        )
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    expected = _signature(config, tokenizer)
    if document["signature"] != expected:
        changed = sorted(
            key for key in expected if document["signature"].get(key) != expected[key]
        )
        raise DataError(
            f"the token cache was built with different settings ({changed}); "
            f"run `embed.py prepare` again"
        )
    sources: dict[str, SourceCache] = {}
    for name, entry in document["sources"].items():
        train = config.data_root / name / TRAIN_FILE
        if not train.is_file() or train.stat().st_size != entry["bytes"]:
            raise DataError(
                f"{train} changed since the token cache was built; run `embed.py prepare` again"
            )
        dev = config.dev.queries_overrides.get(name, config.dev.queries_per_source)
        sources[name] = SourceCache(name, config.cache_root / name, int(dev))
    return TokenCache(config.cache_root, document, sources)


def _signature(config: EmbeddingConfig, tokenizer: TextTokenizer) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "tokenizer": tokenizer.digest,
        "query_prefix": config.text.query_prefix,
        "passage_prefix": config.text.passage_prefix,
        "max_query_tokens": config.text.max_query_tokens,
        "max_passage_tokens": config.text.max_passage_tokens,
        "split_salt": config.dev.split_salt,
    }


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def find_sources(data_root: Path) -> list[str]:
    """Folders under `data_root` with a non-empty train.jsonl, in name order."""
    names = []
    for directory in sorted(Path(data_root).iterdir()):
        train = directory / TRAIN_FILE
        if train.is_file() and train.stat().st_size > 0:
            names.append(directory.name)
    if not names:
        raise DataError(f"no source with a non-empty {TRAIN_FILE} under {data_root}")
    return names


def prepare_cache(config: EmbeddingConfig, workers: int, log=print) -> dict[str, Any]:
    """Tokenise every source with `workers` processes and write the cache."""
    tokenizer = TextTokenizer.from_checkpoint(config.base_checkpoint)
    names = find_sources(config.data_root)
    root = config.cache_root
    scratch = root / ".building"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    jobs = []
    for name in names:
        train = config.data_root / name / TRAIN_FILE
        for index, (start, end) in enumerate(_shards(train)):
            jobs.append((name, index, str(train), start, end, str(scratch / name / f"{index:05d}")))
    # Largest shards first so the pool stays full to the end.
    jobs.sort(key=lambda job: job[3] - job[4])
    log(f"tokenising {len(names)} sources in {len(jobs)} shards with {workers} processes")

    settings = {
        "checkpoint": str(config.base_checkpoint),
        "query_prefix": config.text.query_prefix,
        "passage_prefix": config.text.passage_prefix,
        "max_query_tokens": config.text.max_query_tokens,
        "max_passage_tokens": config.text.max_passage_tokens,
        "split_salt": config.dev.split_salt,
    }
    counts: dict[tuple[str, int], dict[str, int]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_tokenise_shard, job, settings) for job in jobs]
        for done, future in enumerate(futures, start=1):
            name, index, count = future.result()
            counts[(name, index)] = count
            if done % 50 == 0 or done == len(futures):
                log(f"  {done}/{len(futures)} shards")

    entries: dict[str, Any] = {}
    for name in names:
        shards = sorted(index for (source, index) in counts if source == name)
        entry = _merge_source(scratch / name, root / name, shards)
        entry["bytes"] = (config.data_root / name / TRAIN_FILE).stat().st_size
        entry["truncated_queries"] = sum(counts[(name, i)]["truncated_queries"] for i in shards)
        entry["truncated_passages"] = sum(counts[(name, i)]["truncated_passages"] for i in shards)
        entries[name] = entry
        log(
            f"  {name}: {entry['records']:,} records, {entry['passages']:,} passages, "
            f"{entry['query_tokens'] + entry['passage_tokens']:,} tokens"
        )
    shutil.rmtree(scratch)

    signature = _signature(config, tokenizer)
    document = {
        "signature": signature,
        "sources": entries,
        "digest": _digest({"signature": signature, "sources": entries}),
    }
    temporary = root / (CACHE_FILE + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, root / CACHE_FILE)
    return document


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _shards(path: Path) -> list[tuple[int, int]]:
    """Byte ranges of about SHARD_BYTES that each start and end on a line boundary."""
    size = path.stat().st_size
    bounds = [0]
    with path.open("rb") as handle:
        while bounds[-1] < size:
            target = min(bounds[-1] + SHARD_BYTES, size)
            handle.seek(target)
            handle.readline()  # run on to the end of the line that `target` falls in
            bounds.append(min(handle.tell(), size))
    return [(start, end) for start, end in zip(bounds, bounds[1:]) if end > start]


def _tokenise_shard(job: tuple, settings: dict[str, Any]) -> tuple[str, int, dict[str, int]]:
    """Worker: tokenise the records of one byte range and write its part files."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    name, index, train, start, end, output = job
    tokenizer = TextTokenizer.from_checkpoint(settings["checkpoint"])
    max_query, max_passage = settings["max_query_tokens"], settings["max_passage_tokens"]

    queries: list[str] = []
    passages: list[str] = []
    hashes: list[int] = []
    keys: list[int] = []
    positives: list[int] = []
    per_record: list[int] = []
    with open(train, "rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    for line in data.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record["positives"]:
            raise DataError(f"{train}: record {record['id']!r} has no positive")
        queries.append(record["query"])
        keys.append(split_key(settings["split_salt"], record["id"]))
        positives.append(len(record["positives"]))
        per_record.append(len(record["positives"]) + len(record["negatives"]))
        for passage in record["positives"] + record["negatives"]:
            passages.append(format_passage(passage["title"], passage["text"]))
            hashes.append(text_hash(passage["text"]))
    del data

    query_ids = _encode_all(tokenizer, queries, settings["query_prefix"], max_query)
    passage_ids = _encode_all(tokenizer, passages, settings["passage_prefix"], max_passage)

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "query_tokens.npy", _concat(query_ids))
    np.save(directory / "query_lengths.npy", _lengths(query_ids))
    np.save(directory / "passage_tokens.npy", _concat(passage_ids))
    np.save(directory / "passage_lengths.npy", _lengths(passage_ids))
    np.save(directory / "passage_hash.npy", np.asarray(hashes, dtype=np.uint64))
    np.save(directory / "record_passages.npy", np.asarray(per_record, dtype=np.int64))
    np.save(directory / "record_positives.npy", np.asarray(positives, dtype=np.uint16))
    np.save(directory / "split_key.npy", np.asarray(keys, dtype=np.uint64))
    return name, index, {
        "truncated_queries": int(sum(len(ids) == max_query for ids in query_ids)),
        "truncated_passages": int(sum(len(ids) == max_passage for ids in passage_ids)),
    }


def _encode_all(tokenizer: TextTokenizer, texts: list[str], prefix: str, limit: int) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for start in range(0, len(texts), 4096):
        output.extend(tokenizer.encode_batch(texts[start : start + 4096], prefix, limit))
    return output


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=TOKEN_DTYPE)


def _lengths(arrays: list[np.ndarray]) -> np.ndarray:
    return np.asarray([len(array) for array in arrays], dtype=np.int64)


def _merge_source(parts: Path, target: Path, shards: list[int]) -> dict[str, int]:
    """Concatenate the part files of one source, in file order, into the final arrays."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    def gather(name: str) -> np.ndarray:
        return np.concatenate([np.load(parts / f"{i:05d}" / f"{name}.npy") for i in shards])

    def offsets(lengths: np.ndarray) -> np.ndarray:
        output = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=output[1:])
        return output

    arrays = {
        "query_tokens": gather("query_tokens"),
        "query_offsets": offsets(gather("query_lengths")),
        "passage_tokens": gather("passage_tokens"),
        "passage_offsets": offsets(gather("passage_lengths")),
        "passage_hash": gather("passage_hash"),
        "record_start": offsets(gather("record_passages")),
        "record_positives": gather("record_positives"),
        "split_key": gather("split_key"),
    }
    for name, dtype in ARRAYS.items():
        array = arrays[name]
        assert array.dtype == dtype, (name, array.dtype)
        np.save(target / f"{name}.npy", array)
    return {
        "records": int(len(arrays["record_positives"])),
        "passages": int(len(arrays["passage_hash"])),
        "query_tokens": int(len(arrays["query_tokens"])),
        "passage_tokens": int(len(arrays["passage_tokens"])),
    }
