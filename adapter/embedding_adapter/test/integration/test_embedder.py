"""Using a trained adapter: vectors, the decoder flag, late chunking."""

import pytest
import torch

from conftest import cpu_plan, make_config
from src.embedder import TextEmbedder
from src.encoding import encode_tokens
from src.errors import AdapterError, DataError
from src.pooling import hidden_states
from src.trainer import Trainer

QUERIES = ["what is the river of paris alpha one?", "what is the lake of rome beta two?"]
PASSAGES = ["the red river in paris is named alpha one.", "the fox lives on the bank of the lake in rome."]
LONG = " ".join(
    f"the {colour} river in {city} is named {greek}. the {animal} lives on the bank!"
    for colour, city, greek, animal in [
        ("red", "paris", "alpha", "cat"), ("blue", "london", "beta", "dog"), ("green", "rome", "gamma", "bird"),
        ("black", "oslo", "delta", "fox"), ("white", "cairo", "omega", "wolf"), ("yellow", "lima", "sigma", "bear"),
        ("red", "tokyo", "kappa", "lion"), ("blue", "delhi", "theta", "fish"),
    ]
)  # fmt: skip


@pytest.fixture(scope="module")
def trained(workspace, tmp_path_factory):
    config = make_config(workspace)
    run_dir = tmp_path_factory.mktemp("run")
    trainer = Trainer(config, 1, run_dir, cpu_plan(config), log=lambda _: None)
    trainer.train(max_steps=8, evaluate=False)
    trainer._save_adapter(run_dir / "adapter")
    return config, trainer, run_dir / "adapter"


@pytest.fixture(scope="module")
def embedder(trained):
    config, _, adapter = trained
    return TextEmbedder.from_pretrained(config.base_checkpoint, adapter, device="cpu")


def test_a_saved_adapter_gives_the_vectors_of_the_trainer(trained, embedder):
    config, trainer, _ = trained
    tokens = trainer.tokenizer.encode_batch(QUERIES, config.text.query_prefix, config.text.max_query_tokens)
    expected = encode_tokens(trainer.base.encoder, tokens, cpu_plan(config))
    got = embedder.encode_queries(QUERIES)
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got.norm(dim=1), torch.ones(2), atol=1e-6)
    assert embedder.contract["matryoshka_dims"] == [64, 32, 16]


def test_batching_and_order_do_not_change_a_vector(embedder):
    together = embedder.encode_passages(PASSAGES + [LONG])
    alone = embedder.encode_passages([PASSAGES[1]])
    assert torch.allclose(together[1], alone[0], atol=1e-5)
    titled = embedder.encode_passages([PASSAGES[0]], titles=["paris river"])
    assert torch.allclose(titled, embedder.encode_passages(["paris river. " + PASSAGES[0]]), atol=1e-6)
    assert embedder.encode_queries([]).shape == (0, 64)


def test_smaller_dimensions_are_a_renormalised_prefix(embedder):
    full = embedder.encode_queries(QUERIES)
    small = embedder.encode_queries(QUERIES, dim=16)
    assert small.shape == (2, 16)
    assert torch.allclose(small, torch.nn.functional.normalize(full[:, :16], dim=1), atol=1e-6)
    with pytest.raises(DataError, match="was not trained"):
        embedder.encode_queries(QUERIES, dim=24)


def test_the_decoder_is_loaded_only_when_asked_for(trained, embedder):
    config, _, adapter = trained
    assert embedder.base.model is None
    assert all("decoder" not in name for name, _ in embedder.encoder.named_parameters())
    with pytest.raises(AdapterError, match="load_decoder=True"):
        embedder.model
    whole = TextEmbedder.from_pretrained(config.base_checkpoint, adapter, load_decoder=True, device="cpu")
    assert whole.model.encoder is whole.encoder and whole.model.decoder is not None
    assert whole.base.parameter_count() > embedder.base.parameter_count()
    assert torch.equal(whole.encode_queries(QUERIES), embedder.encode_queries(QUERIES))
    assert torch.equal(whole.encode_passages(PASSAGES), embedder.encode_passages(PASSAGES))
    # The adapter sits on the encoder only: the decoder holds no LoRA tensors.
    assert not any("lora_" in name for name, _ in whole.model.decoder.named_parameters())


def test_base_behaviour_switches_the_adapter_off(trained, embedder):
    config, _, _ = trained
    plain = TextEmbedder.from_pretrained(config.base_checkpoint, device="cpu")
    adapted = embedder.encode_queries(QUERIES)
    assert not torch.allclose(adapted, plain.encode_queries(QUERIES), atol=1e-4)
    with embedder.base_behaviour():
        assert torch.equal(embedder.encode_queries(QUERIES), plain.encode_queries(QUERIES))
    assert torch.equal(embedder.encode_queries(QUERIES), adapted)


def test_an_adapter_of_another_base_is_refused(trained, tmp_path):
    from conftest import write_base

    _, _, adapter = trained
    other = write_base(tmp_path / "other", seed=3)
    with pytest.raises(AdapterError, match="different base model"):
        TextEmbedder.from_pretrained(other, adapter, device="cpu")
    with pytest.raises(AdapterError, match="device must be"):
        TextEmbedder.from_pretrained(other, device="tpu")


def test_token_vectors_are_the_final_layer_of_the_encoder(embedder):
    text = PASSAGES[0]
    tokens = embedder.token_vectors(text)
    assert tokens.windows == 1
    encoded = embedder.tokenizer.encode_with_spans(text, "passage: ")
    ids = torch.from_numpy(encoded.ids).unsqueeze(0)
    hidden = hidden_states(embedder.encoder, ids, torch.ones_like(ids), False)[0]
    assert torch.allclose(tokens.vectors, hidden[encoded.content], atol=1e-6)
    assert [text[a:b] for a, b in tokens.spans][:3] == ["the", "red", "river"]
    # The mean over the whole text is the mean of its own tokens.
    assert torch.allclose(tokens.pool(0, len(text)), tokens.vectors.mean(dim=0), atol=1e-6)

    with_title = embedder.token_vectors(text, title="paris river")
    assert len(with_title) == len(tokens)  # the title is context, not content
    assert not torch.allclose(with_title.vectors, tokens.vectors, atol=1e-4)
    with pytest.raises(DataError, match="empty"):
        embedder.token_vectors("  ")


def test_a_long_document_is_covered_by_overlapping_windows(embedder):
    tokens = embedder.token_vectors(LONG)
    assert tokens.windows > 1
    assert len(tokens) == len(embedder.tokenizer.encode_with_spans(LONG, "passage: ").ids) - 3
    assert torch.isfinite(tokens.vectors).all()
    assert (tokens.vectors.abs().sum(dim=1) > 0).all()  # every token got a vector
    # The first tokens come from the first window alone.
    encoded = embedder.tokenizer.encode_with_spans(LONG, "passage: ")
    ids = torch.from_numpy(encoded.ids[:63]).unsqueeze(0)
    ids = torch.cat([ids, torch.tensor([[1]])], dim=1)
    hidden = hidden_states(embedder.encoder, ids, torch.ones_like(ids), False)[0]
    assert torch.allclose(tokens.vectors[:20], hidden[2:22], atol=1e-5)


def test_late_chunks_cover_the_document_and_are_pooled_token_vectors(embedder):
    chunks = embedder.late_chunk(LONG, chunk_tokens=20)
    assert len(chunks) > 3
    assert "".join(chunk.text for chunk in chunks) == LONG
    assert all(chunk.text == LONG[chunk.start : chunk.end] for chunk in chunks)
    assert all(chunk.tokens <= 20 for chunk in chunks)
    tokens = embedder.token_vectors(LONG)
    assert sum(chunk.tokens for chunk in chunks) == len(tokens)
    for chunk in chunks:
        expected = torch.nn.functional.normalize(tokens.pool(chunk.start, chunk.end), dim=0)
        assert torch.allclose(chunk.vector, expected, atol=1e-6)
    # A chunk in context is not the chunk embedded alone: that is the point of late chunking.
    alone = embedder.encode_passages([chunks[2].text])[0]
    assert not torch.allclose(chunks[2].vector, alone, atol=1e-3)

    own = embedder.late_chunk(LONG, spans=[(0, 40), (40, len(LONG))], dim=16)
    assert [chunk.vector.shape[0] for chunk in own] == [16, 16]
    with pytest.raises(DataError, match="outside the text"):
        embedder.late_chunk(LONG, spans=[(0, len(LONG) + 1)])


def test_the_overlap_of_late_chunking_can_be_chosen_per_call(embedder):
    default = embedder.token_vectors(LONG)
    same = embedder.token_vectors(LONG, overlap_tokens=8)  # the adapter's own value
    assert same.windows == default.windows and torch.equal(same.vectors, default.vectors)
    wide = embedder.token_vectors(LONG, overlap_tokens=30)
    none = embedder.token_vectors(LONG, overlap_tokens=0)
    assert none.windows <= default.windows < wide.windows
    for tokens in (wide, none):
        assert len(tokens) == len(default)
        assert torch.isfinite(tokens.vectors).all()
        assert (tokens.vectors.abs().sum(dim=1) > 0).all()  # every token got a vector
    chunks = embedder.late_chunk(LONG, chunk_tokens=20, overlap_tokens=30)
    assert "".join(chunk.text for chunk in chunks) == LONG

    for bad in (-1, 32, 1000):
        with pytest.raises(DataError, match="overlap_tokens must be from 0 to 31"):
            embedder.token_vectors(LONG, overlap_tokens=bad)
    with pytest.raises(DataError, match="whole number"):
        embedder.token_vectors(LONG, overlap_tokens=2.5)
    with pytest.raises(DataError, match="more than twice the overlap"):
        embedder.token_vectors(LONG, overlap_tokens=31)  # the window cannot hold the text then


def test_the_window_of_late_chunking_can_be_chosen_per_call(embedder, monkeypatch):
    default = embedder.token_vectors(LONG)
    same = embedder.token_vectors(LONG, window_tokens=64)  # the adapter's own value
    assert same.windows == default.windows and torch.equal(same.vectors, default.vectors)
    small = embedder.token_vectors(LONG, window_tokens=32)
    big = embedder.token_vectors(LONG, window_tokens=128)
    assert big.windows < default.windows < small.windows
    for tokens in (small, big):
        assert len(tokens) == len(default)
        assert torch.isfinite(tokens.vectors).all()
        assert (tokens.vectors.abs().sum(dim=1) > 0).all()  # every token got a vector
    chunks = embedder.late_chunk(LONG, chunk_tokens=20, window_tokens=96, overlap_tokens=16)
    assert "".join(chunk.text for chunk in chunks) == LONG

    for bad in (16, 31, 129, 1000):
        with pytest.raises(DataError, match="window_tokens must be from 32 to 128"):
            embedder.token_vectors(LONG, window_tokens=bad)
    with pytest.raises(DataError, match="whole number"):
        embedder.token_vectors(LONG, window_tokens=64.0)
    with pytest.raises(DataError, match="overlap_tokens must be from 0 to 15"):
        embedder.token_vectors(LONG, window_tokens=32, overlap_tokens=16)
    monkeypatch.setitem(embedder.contract["late_chunking"], "overlap_tokens", 20)
    with pytest.raises(DataError, match="the adapter's overlap of 20"):
        embedder.token_vectors(LONG, window_tokens=32)  # 20 does not fit a window of 32


def test_the_late_chunking_defaults_can_be_read_and_passed_back(embedder):
    defaults = embedder.late_chunking_defaults
    assert defaults == {"window_tokens": 64, "overlap_tokens": 8, "chunk_tokens": 12}
    defaults["window_tokens"] = 32  # a copy: the adapter's own settings do not change
    assert embedder.late_chunking_defaults["window_tokens"] == 64
    again = embedder.token_vectors(LONG, window_tokens=64, overlap_tokens=8)
    assert torch.equal(again.vectors, embedder.token_vectors(LONG).vectors)
