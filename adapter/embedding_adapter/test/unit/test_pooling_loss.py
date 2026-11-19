"""Pooling, the loss and GradCache."""

import math

import numpy as np
import pytest
import torch

from src.batching import make_chunks
from src.config import AdapterConfig
from src.gradcache import gradcache_step
from src.lora import attach_adapter
from src.loss import contrastive_loss
from src.pooling import mean_pool, normalize, pooled
from test_lora import tiny_encoder


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(mean_pool(hidden, mask), torch.tensor([[2.0, 3.0]]))


def test_vector_does_not_depend_on_padding():
    encoder = tiny_encoder()
    ids = torch.tensor([[5, 9, 12, 30, 1]])
    alone = pooled(encoder, ids, torch.ones_like(ids), bf16=False)
    padded_ids = torch.tensor([[5, 9, 12, 30, 1, 0, 0, 0], [7, 8, 9, 10, 11, 12, 13, 1]])
    together = pooled(encoder, padded_ids, (padded_ids != 0).long(), bf16=False)
    assert torch.allclose(alone[0], together[0], atol=1e-5)


def test_normalize_truncates_then_normalises():
    vectors = torch.tensor([[3.0, 4.0, 12.0]])
    assert torch.allclose(normalize(vectors, 2), torch.tensor([[0.6, 0.8]]))
    assert normalize(vectors).norm().item() == pytest.approx(1.0)


def test_loss_matches_a_hand_computed_example():
    # Unit vectors on the plane; candidate 2 has the same text as query 0's positive.
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    target = torch.tensor([0, 1])
    mask = torch.tensor([[False, False, True], [False, False, False]])
    loss, stats = contrastive_loss(queries, candidates, target, mask, temperature=0.5, dims=[2])
    # query 0: scores (2, 0, masked) -> -log(e^2 / (e^2 + 1))
    # query 1: scores (0, 2, 0)      -> -log(e^2 / (e^2 + 2))
    expected = (math.log(1 + math.exp(-2)) + math.log(1 + 2 * math.exp(-2))) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert stats["accuracy"] == 1.0

    unmasked, _ = contrastive_loss(queries, candidates, target, torch.zeros_like(mask), 0.5, [2])
    assert unmasked.item() > loss.item()  # the duplicate would have been punished as a negative


def test_loss_averages_the_matryoshka_dimensions():
    torch.manual_seed(0)
    queries, candidates = torch.randn(4, 8), torch.randn(6, 8)
    target = torch.arange(4)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    both, stats = contrastive_loss(queries, candidates, target, mask, 0.1, [8, 4])
    full, _ = contrastive_loss(queries, candidates, target, mask, 0.1, [8])
    half, _ = contrastive_loss(queries, candidates, target, mask, 0.1, [4])
    assert both.item() == pytest.approx((full.item() + half.item()) / 2, rel=1e-6)
    assert stats["loss_full"] == pytest.approx(full.item(), rel=1e-6)


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_gradcache_equals_plain_backpropagation(dropout):
    encoder = tiny_encoder()
    adapter = attach_adapter(encoder, AdapterConfig(4, 8.0, True, dropout))
    for module in adapter.modules.values():
        torch.nn.init.normal_(module.lora_B, std=0.05)
    encoder.train()
    rng = np.random.default_rng(0)
    tokens = [np.append(rng.integers(3, 200, size=int(n)), 1) for n in rng.integers(3, 20, size=12)]
    chunks = make_chunks(tokens, budget=60)
    assert len(chunks) > 2
    target = torch.arange(6)
    mask = torch.zeros(6, 6, dtype=torch.bool)

    def encode(chunk):
        return pooled(encoder, chunk.input_ids, chunk.attention_mask, bf16=False)

    def loss_fn(vectors):
        return contrastive_loss(vectors[:6], vectors[6:], target, mask, 0.05, [64, 16])

    loss, _ = gradcache_step(chunks, encode, loss_fn, seed=11)
    cached = [parameter.grad.clone() for parameter in adapter.parameters()]
    for parameter in adapter.parameters():
        parameter.grad = None

    # One graph over every text, each chunk under the seed GradCache gave it.
    vectors = torch.empty(12, 64)
    parts = []
    for number, chunk in enumerate(chunks):
        torch.manual_seed(11 + number)
        parts.append((chunk.index, encode(chunk)))
    vectors = torch.zeros(12, 64).index_put((torch.cat([i for i, _ in parts]),), torch.cat([v for _, v in parts]))
    direct, _ = loss_fn(vectors)
    direct.backward()
    assert loss == pytest.approx(direct.item(), rel=1e-6)
    for got, parameter in zip(cached, adapter.parameters()):
        assert torch.allclose(got, parameter.grad, atol=1e-5, rtol=1e-4)
    assert any(gradient.abs().max() > 0 for gradient in cached)
