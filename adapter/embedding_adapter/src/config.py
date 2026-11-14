"""
Settings of the embedding adapter, read from `embedding_config.yml`.

Loading is strict: an unknown key or a missing one is an error, so a typo in
the YAML can never silently fall back to a default. No training number lives in
Python; this file only says which keys exist and which values are legal.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError

# The projections of one encoder layer that receive a LoRA pair.
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Sources that count as "webfaq" for epochs and for the stage-2 sample; every
# other folder under `data_root` is "curated".
WEBFAQ_SOURCES = ("webfaq-retrieval-english", "webfaq-english-head")


@dataclass(frozen=True)
class AdapterConfig:
    rank: int
    alpha: float
    use_rslora: bool
    dropout: float

    @property
    def scaling(self) -> float:
        return self.alpha / (self.rank**0.5 if self.use_rslora else self.rank)

    def __post_init__(self) -> None:
        _check(self.rank >= 1, "adapter.rank must be at least 1")
        _check(self.alpha > 0, "adapter.alpha must be positive")
        _check(0.0 <= self.dropout < 1.0, "adapter.dropout must be in [0, 1)")


@dataclass(frozen=True)
class TextConfig:
    query_prefix: str
    passage_prefix: str
    max_query_tokens: int
    max_passage_tokens: int
    eval_max_tokens: int

    def __post_init__(self) -> None:
        for name in ("max_query_tokens", "max_passage_tokens", "eval_max_tokens"):
            _check(getattr(self, name) >= 8, f"text.{name} must be at least 8")


@dataclass(frozen=True)
class LossConfig:
    temperature: float
    matryoshka_dims: tuple[int, ...]

    def __post_init__(self) -> None:
        _check(self.temperature > 0, "loss.temperature must be positive")
        _check(len(self.matryoshka_dims) >= 1, "loss.matryoshka_dims must not be empty")
        _check(
            all(d >= 1 for d in self.matryoshka_dims),
            "loss.matryoshka_dims must be positive",
        )


@dataclass(frozen=True)
class DevConfig:
    queries_per_source: int
    queries_overrides: Mapping[str, int]
    split_salt: str
    distractors_per_source: int
    benchmark_tasks: tuple[str, ...]
    benchmark_every: int


@dataclass(frozen=True)
class OptimConfig:
    betas: tuple[float, float]
    weight_decay: float
    grad_clip: float
    precision: str

    def __post_init__(self) -> None:
        _check(
            self.precision in ("auto", "bf16", "fp32"),
            "optim.precision must be auto, bf16 or fp32",
        )
        _check(self.grad_clip > 0, "optim.grad_clip must be positive")


@dataclass(frozen=True)
class StageConfig:
    """One training stage. `hard_negatives == 0` means in-batch negatives only."""

    batch_queries: int
    hard_negatives: int
    lr: float
    warmup_steps: int
    min_lr_ratio: float
    epochs_webfaq: int
    epochs_curated: int
    webfaq_sample: int  # 0 = every WebFAQ record
    source_records: Mapping[str, int]  # per source: at most this many records, drawn once by seed
    source_epochs: Mapping[str, int]  # per source: epochs instead of epochs_webfaq / epochs_curated
    eval_every: int
    save_every: int
    keep_last: int

    def __post_init__(self) -> None:
        _check(self.batch_queries >= 2, "a stage needs at least 2 queries per step")
        _check(self.hard_negatives >= 0, "hard_negatives cannot be negative")
        _check(self.lr > 0, "lr must be positive")
        _check(0.0 <= self.min_lr_ratio <= 1.0, "min_lr_ratio must be in [0, 1]")
        _check(self.save_every >= 1 and self.keep_last >= 1, "save_every/keep_last must be >= 1")
        _check(all(isinstance(v, int) and v >= 1 for v in self.source_records.values()),
               "source_records values must be positive integers")
        _check(all(isinstance(v, int) and v >= 0 for v in self.source_epochs.values()),
               "source_epochs values must be integers >= 0")


@dataclass(frozen=True)
class MiningConfig:
    top_k: int
    skip_top: int
    max_score_ratio: float
    per_query: int

    def __post_init__(self) -> None:
        _check(self.top_k > self.skip_top >= 0, "mining needs top_k > skip_top >= 0")
        _check(self.per_query >= 1, "mining.per_query must be at least 1")


@dataclass(frozen=True)
class LateChunkConfig:
    """Defaults of late chunking; stored with the adapter and used at inference."""

    window_tokens: int  # tokens the encoder sees at once (at most the model's context)
    overlap_tokens: int  # shared by neighbouring windows of a longer document
    chunk_tokens: int  # target size of a chunk made by the built-in splitter

    def __post_init__(self) -> None:
        _check(self.window_tokens >= 32, "late_chunking.window_tokens is too small")
        _check(
            0 <= self.overlap_tokens < self.window_tokens // 2,
            "late_chunking.overlap_tokens must be below half the window",
        )
        _check(self.chunk_tokens >= 4, "late_chunking.chunk_tokens is too small")


@dataclass(frozen=True)
class ResourceConfig:
    gpu_memory_gb: float  # cap for this process; 0 = no cap
    chunk_tokens: int  # padded tokens per GradCache chunk (with gradients)
    encode_tokens: int  # padded tokens per batch without gradients
    cpu_threads: int  # 0 = chosen by resources.plan_device
    cpu_chunk_tokens: int
    seed: int

    def __post_init__(self) -> None:
        _check(self.chunk_tokens >= 256, "resources.chunk_tokens is too small")
        _check(self.encode_tokens >= 256, "resources.encode_tokens is too small")
        _check(self.cpu_chunk_tokens >= 256, "resources.cpu_chunk_tokens is too small")


@dataclass(frozen=True)
class EmbeddingConfig:
    base_checkpoint: Path
    data_root: Path
    cache_root: Path
    eval_root: Path
    adapter: AdapterConfig
    text: TextConfig
    loss: LossConfig
    dev: DevConfig
    optim: OptimConfig
    stage_1: StageConfig
    stage_2: StageConfig
    mining: MiningConfig
    late_chunking: LateChunkConfig
    resources: ResourceConfig

    def stage(self, number: int) -> StageConfig:
        if number == 1:
            return self.stage_1
        if number == 2:
            return self.stage_2
        raise ConfigError(f"there is no stage {number}; stages are 1 and 2")

    def to_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))

    def digest(self, *sections: str) -> str:
        """Hash of the named top-level sections (all of them when none is named)."""
        data = self.to_dict()
        picked = {name: data[name] for name in sections} if sections else data
        text = json.dumps(picked, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> EmbeddingConfig:
    """Read the YAML at `path`; `overrides` maps dotted keys (`adapter.rank`) to values."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must hold a mapping at the top level")
    for dotted, value in (overrides or {}).items():
        _set_dotted(raw, dotted, value)
    return build_config(raw)


def build_config(raw: Mapping[str, Any]) -> EmbeddingConfig:
    top = _section(raw, "config", EmbeddingConfig)
    dev = dict(_mapping(top["dev"], "dev"))
    dev.setdefault("queries_overrides", {})
    stages = {}
    for name in ("stage_1", "stage_2"):
        stages[name] = dict(_mapping(top[name], name))
        stages[name].setdefault("source_records", {})
        stages[name].setdefault("source_epochs", {})
    return EmbeddingConfig(
        base_checkpoint=Path(top["base_checkpoint"]),
        data_root=Path(top["data_root"]),
        cache_root=Path(top["cache_root"]),
        eval_root=Path(top["eval_root"]),
        adapter=_build(AdapterConfig, top["adapter"], "adapter"),
        text=_build(TextConfig, top["text"], "text"),
        loss=_build(LossConfig, top["loss"], "loss"),
        dev=_build(DevConfig, dev, "dev"),
        optim=_build(OptimConfig, top["optim"], "optim"),
        stage_1=_build(StageConfig, stages["stage_1"], "stage_1"),
        stage_2=_build(StageConfig, stages["stage_2"], "stage_2"),
        mining=_build(MiningConfig, top["mining"], "mining"),
        late_chunking=_build(LateChunkConfig, top["late_chunking"], "late_chunking"),
        resources=_build(ResourceConfig, top["resources"], "resources"),
    )


def _build(cls: type, raw: Any, name: str):
    values = dict(_section(_mapping(raw, name), name, cls))
    for field in dataclasses.fields(cls):
        value = values[field.name]
        kind = str(field.type)
        if kind.startswith("tuple"):
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"{name}.{field.name} must be a list")
            values[field.name] = tuple(value)
        elif kind == "float" and isinstance(value, int) and not isinstance(value, bool):
            values[field.name] = float(value)
        elif kind in ("int", "float", "bool", "str"):
            expected = {"int": int, "float": float, "bool": bool, "str": str}[kind]
            if not isinstance(value, expected) or (kind == "int" and isinstance(value, bool)):
                raise ConfigError(
                    f"{name}.{field.name} must be {kind}, got {value!r}"
                )
    return cls(**values)


def _section(raw: Mapping[str, Any], name: str, cls: type) -> Mapping[str, Any]:
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    missing = sorted(known - set(raw))
    if unknown:
        raise ConfigError(f"unknown key(s) in {name}: {unknown}")
    if missing:
        raise ConfigError(f"missing key(s) in {name}: {missing}")
    return raw


def _mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return raw


def _set_dotted(raw: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = raw
    for key in parents:
        if not isinstance(node.get(key), dict):
            raise ConfigError(f"cannot override {dotted!r}: {key!r} is not a section")
        node = node[key]
    if leaf not in node:
        raise ConfigError(f"cannot override {dotted!r}: no such key")
    node[leaf] = value


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)
