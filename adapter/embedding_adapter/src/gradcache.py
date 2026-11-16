"""
GradCache (Gao et al. 2021): a contrastive step over more texts than fit in
memory with gradients.

  1. Encode every chunk without gradients and keep the vectors.
  2. Compute the loss on the vectors and backpropagate to the vectors only.
  3. Re-encode each chunk with gradients and backpropagate
     sum(vector * cached gradient of that vector).

The parameter gradients are those of one big batch (tested against plain
backpropagation). Dropout must draw the same mask in pass 1 and pass 3, so each
chunk is encoded under its own seed in both.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from .batching import Chunk

# encode(chunk) -> un-normalised vectors [n, D], float32, in the chunk's row order
EncodeFn = Callable[[Chunk], torch.Tensor]
# loss(vectors [texts, D]) -> (scalar loss, stats)
LossFn = Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float]]]


def gradcache_step(
    chunks: Sequence[Chunk], encode: EncodeFn, loss_fn: LossFn, seed: int
) -> tuple[float, dict[str, float]]:
    """Accumulate the step's gradients into the trainable parameters; return the loss."""
    texts = sum(len(chunk.index) for chunk in chunks)

    vectors: torch.Tensor | None = None
    with torch.no_grad():
        for number, chunk in enumerate(chunks):
            torch.manual_seed(seed + number)
            output = encode(chunk)
            if vectors is None:
                vectors = output.new_empty((texts, output.shape[1]))
            vectors[chunk.index] = output
    assert vectors is not None

    vectors.requires_grad_(True)
    loss, stats = loss_fn(vectors)
    (gradient,) = torch.autograd.grad(loss, vectors)

    for number, chunk in enumerate(chunks):
        torch.manual_seed(seed + number)
        output = encode(chunk)
        (output * gradient[chunk.index]).sum().backward()

    return float(loss.detach()), stats
