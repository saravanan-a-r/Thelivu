"""
Loading the base model: the encoder alone, or the whole network.

`load_decoder=False` (the default for embedding work) builds the tied token
embedding and the encoder stack and reads only their tensors from
`model.safetensors`; the decoder is never constructed, read or moved to the
device. That is about 40% of the network instead of all of it.

`load_decoder=True` builds the complete encoder-decoder, for a process that
also generates text (for example an answering adapter that reads the chunks
this adapter retrieved). The encoder object is the same either way, so the
embedding code does not care which mode was used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from .errors import AdapterError
from .interop import base_model_pkg as base_pkg

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


@dataclass
class BaseModel:
    """The loaded base. `model` and `decoder` are None in encoder-only mode."""

    config: object  # thelivu_model.ModelConfig
    encoder: nn.Module
    model: nn.Module | None
    fingerprint: str
    checkpoint: Path

    @property
    def decoder(self) -> nn.Module | None:
        return None if self.model is None else self.model.decoder

    @property
    def has_decoder(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> torch.device:
        return self.encoder.embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.embedding.weight.dtype

    def parameter_count(self) -> int:
        module = self.model if self.model is not None else self.encoder
        return sum(parameter.numel() for parameter in module.parameters())


def read_model_config(checkpoint: str | Path):
    path = Path(checkpoint) / CONFIG_FILE
    if not path.is_file():
        raise AdapterError(f"no {CONFIG_FILE} in base checkpoint {checkpoint}")
    with path.open("r", encoding="utf-8") as handle:
        config = base_pkg.model_config_from_huggingface(json.load(handle))
    # Inference and adapter training both run the base without its own dropout
    # or label smoothing; the adapter brings its own dropout.
    return config.with_(dropout_rate=0.0, label_smoothing=0.0, expected_parameters=None)


def load_base(
    checkpoint: str | Path,
    *,
    load_decoder: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BaseModel:
    """Load the base from a release folder (`config.json` + `model.safetensors`)."""
    checkpoint = Path(checkpoint)
    weights = checkpoint / WEIGHTS_FILE
    if not weights.is_file():
        raise AdapterError(f"no {WEIGHTS_FILE} in base checkpoint {checkpoint}")
    config = read_model_config(checkpoint)

    if load_decoder:
        model = base_pkg.build_model(config, verify=False)
        state = _read_tensors(weights, encoder_only=False)
        base_pkg.load_huggingface_state_dict(model, state)
        encoder = model.encoder
        root: nn.Module = model
    else:
        model = None
        encoder = build_encoder(config)
        _load_encoder_tensors(encoder, _read_tensors(weights, encoder_only=True))
        root = encoder

    root.requires_grad_(False)
    root.eval()
    root.to(device=device, dtype=dtype)
    return BaseModel(
        config=config,
        encoder=encoder,
        model=model,
        fingerprint=encoder_fingerprint(encoder),
        checkpoint=checkpoint,
    )


def build_encoder(config) -> nn.Module:
    """An encoder with its own token embedding, the way the full model builds it."""
    embedding = nn.Embedding(config.vocab_size, config.d_model)
    return base_pkg.Encoder(config, embedding)


def _read_tensors(weights: Path, *, encoder_only: bool) -> dict[str, torch.Tensor]:
    """Read tensors under their HuggingFace names; skip the decoder when asked to."""
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if encoder_only and not (name == "shared.weight" or name.startswith("encoder.")):
                continue
            state[name] = handle.get_tensor(name)
    return state


def _load_encoder_tensors(encoder: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Strict load of `shared.weight` + `encoder.*` into a stand-alone encoder."""
    translated: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        local = base_pkg.from_huggingface_name(name)
        if local == "shared.weight":
            translated["embedding.weight"] = tensor
        else:
            translated[local.removeprefix("encoder.")] = tensor
    own = encoder.state_dict()
    missing = sorted(set(own) - set(translated))
    unexpected = sorted(set(translated) - set(own))
    if missing or unexpected:
        raise AdapterError(
            f"the checkpoint does not match the encoder: missing {missing[:3]} "
            f"({len(missing)}), unexpected {unexpected[:3]} ({len(unexpected)})"
        )
    for name, tensor in translated.items():
        if tuple(tensor.shape) != tuple(own[name].shape):
            raise AdapterError(
                f"{name}: checkpoint has shape {tuple(tensor.shape)}, "
                f"the encoder needs {tuple(own[name].shape)}"
            )
    encoder.load_state_dict(translated, strict=True)


def encoder_fingerprint(encoder: nn.Module) -> str:
    """
    Identity of the encoder an adapter was trained on.

    SHA-256 over the token embedding, the relative position table and every
    normalisation gain of the encoder. These are present in both load modes, so
    an adapter trained on the encoder alone is accepted by the full model and
    the other way round. LoRA tensors are not part of it.
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(encoder.named_parameters()):
        if "lora_" in name:
            continue
        if name == "embedding.weight" or name.startswith("position_bias.") or "norm" in name:
            digest.update(name.encode("utf-8"))
            digest.update(
                parameter.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
            )
    return digest.hexdigest()


def full_base_hash(encoder: nn.Module) -> str:
    """SHA-256 over every base tensor of the encoder (slow; used to prove nothing moved)."""
    # A wrapped projection keeps its weight under `<name>.base.weight`; hash it under
    # the name it had before, so the value is the same with and without an adapter.
    base = {
        name.replace(".base.weight", ".weight"): parameter
        for name, parameter in encoder.named_parameters()
        if "lora_" not in name
    }
    digest = hashlib.sha256()
    for name in sorted(base):
        digest.update(name.encode("utf-8"))
        digest.update(base[name].detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()
