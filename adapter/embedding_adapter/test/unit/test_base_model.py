"""Encoder-only and whole-model loading give the same encoder."""

import pytest
import torch

from src.base_model import full_base_hash, load_base
from src.errors import AdapterError


def test_encoder_only_load_has_no_decoder(config):
    base = load_base(config.base_checkpoint, load_decoder=False)
    assert base.model is None and base.decoder is None and not base.has_decoder
    assert all(not name.startswith("decoder") for name, _ in base.encoder.named_parameters())
    assert not any(parameter.requires_grad for parameter in base.encoder.parameters())
    assert not base.encoder.training


def test_both_modes_give_the_same_encoder(config):
    alone = load_base(config.base_checkpoint, load_decoder=False)
    whole = load_base(config.base_checkpoint, load_decoder=True)
    assert whole.has_decoder and whole.decoder is whole.model.decoder
    assert whole.encoder is whole.model.encoder
    assert alone.fingerprint == whole.fingerprint
    assert full_base_hash(alone.encoder) == full_base_hash(whole.encoder)
    assert alone.parameter_count() < whole.parameter_count()
    ids = torch.tensor([[5, 9, 12, 30, 1, 0], [7, 8, 9, 10, 11, 1]])
    mask = (ids != 0).long()
    with torch.no_grad():
        assert torch.equal(alone.encoder(ids, mask), whole.encoder(ids, mask))


def test_a_different_base_has_a_different_fingerprint(config, tmp_path):
    from conftest import write_base

    other = load_base(write_base(tmp_path / "other", seed=1))
    assert other.fingerprint != load_base(config.base_checkpoint).fingerprint


def test_missing_files_are_reported(tmp_path):
    with pytest.raises(AdapterError, match="model.safetensors"):
        load_base(tmp_path)
