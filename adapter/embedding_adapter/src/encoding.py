"""Encoding many tokenised texts without gradients, in length-sorted batches."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn

from .batching import make_chunks
from .errors import DataError
from .pooling import normalize, pooled


class EncodePlan(Protocol):
    """What encoding needs to know about the device (`resources.DevicePlan` fits)."""

    device: torch.device
    bf16: bool
    encode_tokens: int


@torch.no_grad()
def encode_tokens(
    encoder: nn.Module, tokens: Sequence[np.ndarray], plan: EncodePlan, *, unit: bool = True
) -> torch.Tensor:
    """
    Text vectors [texts, d_model], float32, on the plan's device, in input order.

    Unit length by default; `unit=False` returns the plain mean of the token vectors.
    """
    if len(tokens) == 0:
        raise DataError("nothing to encode")
    was_training = encoder.training
    encoder.eval()
    try:
        output: torch.Tensor | None = None
        for chunk in make_chunks(list(tokens), plan.encode_tokens):
            chunk = chunk.to(plan.device)
            vectors = pooled(encoder, chunk.input_ids, chunk.attention_mask, plan.bf16)
            if unit:
                vectors = normalize(vectors)
            if output is None:
                output = vectors.new_empty((len(tokens), vectors.shape[1]))
            output[chunk.index] = vectors
        assert output is not None
        return output
    finally:
        encoder.train(was_training)
