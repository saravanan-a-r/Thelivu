"""
Saving and loading an embedding adapter.

An adapter folder holds

    adapter.safetensors    the LoRA tensors, float32
    adapter_config.json    rank/alpha/dropout, the text and pooling contract,
                           and the fingerprint of the encoder it was trained on

Loading refuses an adapter whose fingerprint differs from the encoder it is
being attached to: the vectors would be meaningless.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn
from safetensors.torch import load_file, save_file

from .base_model import BaseModel
from .config import AdapterConfig, EmbeddingConfig
from .errors import AdapterError
from .lora import EncoderAdapter, attach_adapter

ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
FORMAT = "thelivu-embedding-adapter"
FORMAT_VERSION = 1


def embedding_contract(config: EmbeddingConfig) -> dict[str, Any]:
    """What a caller must do to get the vectors the adapter was trained to give."""
    return {
        "query_prefix": config.text.query_prefix,
        "passage_prefix": config.text.passage_prefix,
        "max_query_tokens": config.text.max_query_tokens,
        "max_passage_tokens": config.text.max_passage_tokens,
        "eval_max_tokens": config.text.eval_max_tokens,
        "append_eos": True,
        "pooling": "mean",
        "normalize": True,
        "matryoshka_dims": list(config.loss.matryoshka_dims),
        "late_chunking": dataclasses.asdict(config.late_chunking),
    }


def save_adapter(
    directory: str | Path,
    adapter: EncoderAdapter,
    base: BaseModel,
    contract: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(adapter.state_dict(), str(directory / ADAPTER_WEIGHTS))
    document = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "adapter": dataclasses.asdict(adapter.config),
        "parameters": adapter.parameter_count(),
        "base_fingerprint": base.fingerprint,
        "d_model": base.config.d_model,
        "embedding": dict(contract),
        "extra": dict(extra or {}),
    }
    _write_json(directory / ADAPTER_CONFIG, document)
    return directory


def read_adapter_config(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / ADAPTER_CONFIG
    if not path.is_file():
        raise AdapterError(f"no {ADAPTER_CONFIG} in {directory}")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("format") != FORMAT:
        raise AdapterError(f"{path} is not an embedding adapter (format {document.get('format')!r})")
    if document.get("format_version") != FORMAT_VERSION:
        raise AdapterError(
            f"{path} has format version {document.get('format_version')}; "
            f"this code reads version {FORMAT_VERSION}"
        )
    return document


def load_adapter(directory: str | Path, base: BaseModel) -> tuple[EncoderAdapter, dict[str, Any]]:
    """Attach the adapter stored in `directory` to `base.encoder`."""
    directory = Path(directory)
    document = read_adapter_config(directory)
    if document["base_fingerprint"] != base.fingerprint:
        raise AdapterError(
            f"the adapter in {directory} was trained on a different base model "
            f"(adapter {document['base_fingerprint'][:12]}, base {base.fingerprint[:12]})"
        )
    adapter = attach_adapter(base.encoder, AdapterConfig(**document["adapter"]))
    adapter.load_state_dict(load_file(str(directory / ADAPTER_WEIGHTS)))
    if adapter.parameter_count() != document["parameters"]:
        raise AdapterError(
            f"adapter has {adapter.parameter_count():,} parameters, "
            f"its config says {document['parameters']:,}"
        )
    _eval_mode(base.encoder)
    return adapter, document


def _eval_mode(module: nn.Module) -> None:
    module.eval()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)
