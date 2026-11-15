"""
LoRA pairs on the frozen projections of the encoder.

Each targeted projection `W` (frozen, bias-free) becomes

    y = W x + scaling * B (A dropout(x))        A: (rank, in)   B: (out, rank)

`B` starts at zero, so a freshly attached adapter is an exact no-op. `scaling`
is `alpha / sqrt(rank)` with rsLoRA, else `alpha / rank`.

The pairs are never merged into the base weights. That keeps the base
bit-identical (one base checkpoint serves every adapter) and lets a caller
switch the adapter off for a while (`EncoderAdapter.disabled()`), which the
full-model mode needs: a decoder-side adapter may want the plain encoder.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ATTENTION_PROJECTIONS, FFN_PROJECTIONS, AdapterConfig
from .errors import AdapterError


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, scaling: float, dropout: float) -> None:
        super().__init__()
        if base.bias is not None:
            raise AdapterError("the base model's projections are bias-free; got one with a bias")
        self.base = base
        self.rank = rank
        self.scaling = scaling
        self.enabled = True
        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=weight.dtype, device=weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=weight.dtype, device=weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        """The frozen base weight, for code that inspects `module.weight`."""
        return self.base.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base(hidden_states)
        if not self.enabled:
            return output
        update = F.linear(F.linear(self.dropout(hidden_states), self.lora_A), self.lora_B)
        return output + update * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.4f}, enabled={self.enabled}"


class EncoderAdapter:
    """Handle on every LoRA pair attached to one encoder."""

    def __init__(self, config: AdapterConfig, modules: dict[str, LoRALinear]) -> None:
        self.config = config
        self.modules = modules

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, module in self.modules.items():
            yield f"{name}.lora_A", module.lora_A
            yield f"{name}.lora_B", module.lora_B

    def parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Float32 CPU copies, whatever dtype and device the model runs on."""
        return {
            name: parameter.detach().to("cpu", torch.float32).clone()
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        own = dict(self.named_parameters())
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise AdapterError(
                f"adapter tensors do not match this encoder: "
                f"{len(missing)} missing (first: {missing[:2]}), "
                f"{len(unexpected)} unexpected (first: {unexpected[:2]})"
            )
        with torch.no_grad():
            for name, parameter in own.items():
                tensor = state[name]
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise AdapterError(
                        f"{name}: adapter has shape {tuple(tensor.shape)}, "
                        f"this encoder needs {tuple(parameter.shape)}"
                    )
                parameter.copy_(tensor.to(parameter.device, parameter.dtype))

    def set_enabled(self, enabled: bool) -> None:
        for module in self.modules.values():
            module.enabled = enabled

    @contextmanager
    def disabled(self) -> Iterator[None]:
        """Run the plain base encoder inside the block."""
        previous = [module.enabled for module in self.modules.values()]
        self.set_enabled(False)
        try:
            yield
        finally:
            for module, state in zip(self.modules.values(), previous):
                module.enabled = state


def attach_adapter(encoder: nn.Module, config: AdapterConfig) -> EncoderAdapter:
    """
    Wrap every attention and feed-forward projection of `encoder` in place.

    Freezes every parameter the encoder already has (the tied embedding
    included); afterwards only the LoRA pairs require gradients.
    """
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    modules: dict[str, LoRALinear] = {}
    for index, layer in enumerate(encoder.layers):
        for owner_name, projections in (
            ("self_attn", ATTENTION_PROJECTIONS),
            ("ffn", FFN_PROJECTIONS),
        ):
            owner = getattr(layer, owner_name)
            for projection in projections:
                base = getattr(owner, projection)
                if isinstance(base, LoRALinear):
                    raise AdapterError("this encoder already has an adapter attached")
                wrapped = LoRALinear(base, config.rank, config.scaling, config.dropout)
                setattr(owner, projection, wrapped)
                modules[f"encoder.layers.{index}.{owner_name}.{projection}"] = wrapped
    return EncoderAdapter(config, modules)


def expected_adapter_parameters(model_config, config: AdapterConfig) -> int:
    """Parameter count `attach_adapter` must produce, computed from the shapes alone."""
    inner = model_config.num_heads * model_config.d_kv
    d_model, d_ff = model_config.d_model, model_config.d_ff
    attention = 4 * (d_model + inner)  # q, k, v: d_model -> inner; o: inner -> d_model
    ffn = 3 * (d_model + d_ff)  # gate, up: d_model -> d_ff; down: d_ff -> d_model
    return model_config.num_layers * config.rank * (attention + ffn)
