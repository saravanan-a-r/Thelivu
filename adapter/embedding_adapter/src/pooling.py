"""
Encoder output to vectors.

The encoder may run in bf16; everything from the hidden states on is float32,
so a vector does not depend on the precision the encoder ran in more than the
encoder itself does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_states(
    encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, bf16: bool
) -> torch.Tensor:
    """Final-layer token vectors (after the final norm), float32 [n, length, d_model]."""
    if bf16:
        with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16):
            output = encoder(input_ids, attention_mask)
    else:
        output = encoder(input_ids, attention_mask)
    return output.float()


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over the real tokens of each text; padding does not count."""
    weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def pooled(
    encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, bf16: bool
) -> torch.Tensor:
    """Un-normalised text vectors, float32 [n, d_model]."""
    return mean_pool(hidden_states(encoder, input_ids, attention_mask, bf16), attention_mask)


def normalize(vectors: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Unit vectors; `dim` keeps the first `dim` components (Matryoshka) before normalising."""
    if dim is not None:
        vectors = vectors[..., :dim]
    return F.normalize(vectors, dim=-1)
