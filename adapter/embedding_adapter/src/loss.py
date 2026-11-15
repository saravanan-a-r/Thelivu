"""
Masked InfoNCE with Matryoshka dimensions.

    s(i, c) = cos(q_i, c) / temperature          over every candidate of the step
    loss_d  = cross_entropy(s, target = the positive of q_i)   on vectors cut to d
    loss    = mean of loss_d over the Matryoshka dimensions

`mask[i, c]` is True where candidate `c` must not count as a negative for
query `i` (its text is one of the query's own positives). Those scores are
removed before the softmax.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def contrastive_loss(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    dims: Sequence[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    `queries` [Q, D] and `candidates` [C, D] are un-normalised pooled vectors.
    Returns the loss and, for logging, the full-dimension loss and top-1 accuracy.
    """
    losses = []
    stats: dict[str, float] = {}
    for dim in dims:
        q = F.normalize(queries[:, :dim], dim=-1)
        c = F.normalize(candidates[:, :dim], dim=-1)
        scores = (q @ c.T) / temperature
        scores = scores.masked_fill(mask, float("-inf"))
        losses.append(F.cross_entropy(scores, target))
        if dim == max(dims):
            stats["loss_full"] = float(losses[-1].detach())
            stats["accuracy"] = float((scores.argmax(dim=1) == target).float().mean())
    return torch.stack(losses).mean(), stats
