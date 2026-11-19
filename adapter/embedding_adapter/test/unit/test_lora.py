"""The adapter: size, what it trains, and that it starts as a no-op."""

import pytest
import torch

from conftest import REAL_CONFIG, TINY_MODEL
from src.base_model import build_encoder, full_base_hash, read_model_config
from src.config import AdapterConfig, load_config
from src.errors import AdapterError
from src.interop import base_model_pkg
from src.lora import attach_adapter, expected_adapter_parameters


def tiny_encoder(seed: int = 0):
    torch.manual_seed(seed)
    # Frozen like a loaded base: PyTorch picks other kernels for tensors that need
    # gradients, and the comparisons below are bit-exact.
    return build_encoder(base_model_pkg.ModelConfig(**TINY_MODEL)).eval().requires_grad_(False)


def inputs():
    ids = torch.tensor([[5, 9, 12, 30, 1, 0, 0], [7, 8, 9, 10, 11, 12, 1]])
    return ids, (ids != 0).long()


def test_parameter_count_matches_the_formula():
    encoder = tiny_encoder()
    config = AdapterConfig(rank=4, alpha=8.0, use_rslora=True, dropout=0.0)
    adapter = attach_adapter(encoder, config)
    model_config = base_model_pkg.ModelConfig(**TINY_MODEL)
    assert adapter.parameter_count() == expected_adapter_parameters(model_config, config)
    # 2 layers x [4 attention projections (64 <-> 64) + gate, up (64 -> 176) + down (176 -> 64)]
    per_layer = 4 * 4 * (64 + 64) + 3 * 4 * (64 + 176)
    assert adapter.parameter_count() == 2 * per_layer
    assert len(adapter.modules) == 2 * 7


def test_released_base_at_rank_32_has_15_138_816_parameters():
    config = load_config(REAL_CONFIG)
    if not (config.base_checkpoint / "config.json").is_file():
        pytest.skip("released base model not on this machine")
    model_config = read_model_config(config.base_checkpoint)
    assert expected_adapter_parameters(model_config, config.adapter) == 15_138_816


def test_only_lora_tensors_require_gradients():
    encoder = tiny_encoder()
    adapter = attach_adapter(encoder, AdapterConfig(4, 8.0, True, 0.0))
    trainable = {name for name, parameter in encoder.named_parameters() if parameter.requires_grad}
    assert trainable and all(".lora_A" in name or ".lora_B" in name for name in trainable)
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) == adapter.parameter_count()
    assert not encoder.embedding.weight.requires_grad


def test_new_adapter_leaves_the_output_unchanged():
    encoder = tiny_encoder()
    ids, mask = inputs()
    before = encoder(ids, mask)
    attach_adapter(encoder, AdapterConfig(4, 8.0, True, 0.0))
    assert torch.equal(encoder(ids, mask), before)


def test_disabled_gives_the_base_output_and_restores():
    encoder = tiny_encoder()
    ids, mask = inputs()
    before = encoder(ids, mask)
    adapter = attach_adapter(encoder, AdapterConfig(4, 8.0, True, 0.0))
    for module in adapter.modules.values():
        torch.nn.init.normal_(module.lora_B, std=0.1)
    changed = encoder(ids, mask)
    assert not torch.allclose(changed, before, atol=1e-4)
    with adapter.disabled():
        assert torch.equal(encoder(ids, mask), before)
    assert torch.equal(encoder(ids, mask), changed)


def test_rslora_scaling():
    assert AdapterConfig(32, 8.0, True, 0.0).scaling == pytest.approx(8 / 32**0.5)
    assert AdapterConfig(32, 8.0, False, 0.0).scaling == pytest.approx(8 / 32)


def test_state_round_trip_and_strict_loading():
    encoder = tiny_encoder()
    adapter = attach_adapter(encoder, AdapterConfig(4, 8.0, True, 0.0))
    for module in adapter.modules.values():
        torch.nn.init.normal_(module.lora_B, std=0.1)
    state = adapter.state_dict()
    other_encoder = tiny_encoder()
    other = attach_adapter(other_encoder, AdapterConfig(4, 8.0, True, 0.0))
    other.load_state_dict(state)
    ids, mask = inputs()
    assert torch.equal(other_encoder(ids, mask), encoder(ids, mask))

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(AdapterError):
        other.load_state_dict(missing)
    wrong = attach_adapter(tiny_encoder(), AdapterConfig(8, 8.0, True, 0.0))
    with pytest.raises(AdapterError):
        wrong.load_state_dict(state)


def test_attaching_twice_is_refused_and_base_is_untouched():
    encoder = tiny_encoder()
    before = full_base_hash(encoder)
    attach_adapter(encoder, AdapterConfig(4, 8.0, True, 0.0))
    assert full_base_hash(encoder) == before
    with pytest.raises(AdapterError):
        attach_adapter(encoder, AdapterConfig(4, 8.0, True, 0.0))
