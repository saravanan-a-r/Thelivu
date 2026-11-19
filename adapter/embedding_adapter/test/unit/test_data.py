"""Token cache, step plan and batching."""

import json

import numpy as np
import pytest

from conftest import make_config, make_records, write_source
from src.batching import build_step, chunk_order, make_chunks, padded_tokens, step_passages
from src.cache import open_cache, prepare_cache
from src.errors import DataError
from src.plan import build_plan, positive_choice, stage_records
from src.text import TextTokenizer, format_passage, text_hash


@pytest.fixture
def tokenizer(config):
    return TextTokenizer.from_checkpoint(config.base_checkpoint)


@pytest.fixture
def cache(config, tokenizer):
    return open_cache(config, tokenizer)


def test_cache_holds_exactly_the_tokenised_file(config, cache, tokenizer):
    assert cache.names == ["hotpotqa", "squad", "webfaq-retrieval-english"]  # the empty source is left out
    for name in cache.names:
        source = cache[name]
        rows = [json.loads(line) for line in (config.data_root / name / "train.jsonl").open()]
        assert source.records == len(rows)
        passage = 0
        for index, row in enumerate(rows):
            expected = tokenizer.encode_batch([row["query"]], "query: ", config.text.max_query_tokens)[0]
            assert np.array_equal(source.query(index), expected)
            assert source.record_start[index] == passage
            assert source.record_positives[index] == len(row["positives"])
            for item in row["positives"] + row["negatives"]:
                text = format_passage(item["title"], item["text"])
                expected = tokenizer.encode_batch([text], "passage: ", config.text.max_passage_tokens)[0]
                assert np.array_equal(source.passage(passage), expected)
                assert int(source.passage_hash[passage]) == text_hash(item["text"])
                passage += 1
        assert source.passages == passage


def test_every_text_ends_with_eos_and_respects_the_limit(config, cache):
    for name in cache.names:
        source = cache[name]
        assert source.query_lengths.max() <= config.text.max_query_tokens
        assert source.passage_lengths.max() <= config.text.max_passage_tokens
        assert all(source.query(r)[-1] == 1 for r in range(source.records))
        assert all(source.passage(p)[-1] == 1 for p in range(source.passages))


def test_dev_split_is_fixed_disjoint_and_sized(config, cache):
    assert len(cache["squad"].dev_records) == 4  # the override
    assert len(cache["hotpotqa"].dev_records) == 6
    for name in cache.names:
        source = cache[name]
        assert not set(source.dev_records) & set(source.train_records)
        assert len(source.dev_records) + len(source.train_records) == source.records


def test_cache_refuses_changed_settings_or_data(workspace, tmp_path, tokenizer):
    changed = make_config(workspace, text={"max_passage_tokens": 40})
    with pytest.raises(DataError, match="different settings"):
        open_cache(changed, tokenizer)

    # A private copy of one source, then the file grows.
    import shutil

    shutil.copytree(workspace / "base", tmp_path / "base")
    write_source(tmp_path / "data", "squad", make_records("squad", 20))
    own = make_config(tmp_path)
    prepare_cache(own, workers=1, log=lambda _: None)
    open_cache(own, tokenizer)
    write_source(tmp_path / "data", "squad", make_records("squad", 21))
    with pytest.raises(DataError, match="changed since"):
        open_cache(own, tokenizer)


def test_plan_is_deterministic_single_source_and_never_uses_dev(config, cache):
    for stage_number in (1, 2):
        stage = config.stage(stage_number)
        plan = build_plan(cache, stage, stage_number, config.resources.seed)
        again = build_plan(cache, stage, stage_number, config.resources.seed)
        assert plan.digest == again.digest
        assert np.array_equal(plan.records, again.records)
        assert build_plan(cache, stage, stage_number, config.resources.seed + 1).digest != plan.digest
        for index in range(plan.steps):
            name, epoch, records = plan.step(index)
            source = cache[name]
            assert len(records) == stage.batch_queries
            assert len(set(records.tolist())) == len(records)
            assert not set(records.tolist()) & set(source.dev_records.tolist())
            assert set(records.tolist()) <= set(stage_records(source, stage, cache, config.resources.seed).tolist())


def test_plan_visits_each_record_at_most_once_per_epoch(config, cache):
    plan = build_plan(cache, config.stage_1, 1, config.resources.seed)
    seen: dict[tuple[str, int], list[int]] = {}
    for index in range(plan.steps):
        name, epoch, records = plan.step(index)
        seen.setdefault((name, epoch), []).extend(records.tolist())
    assert {name for name, _ in seen} == set(cache.names)
    for (name, epoch), records in seen.items():
        assert len(records) == len(set(records))
        full = len(cache[name].train_records) // config.stage_1.batch_queries * config.stage_1.batch_queries
        assert len(records) == full
    assert sum(1 for name, _ in seen if name == "squad") == config.stage_1.epochs_curated
    assert sum(1 for name, _ in seen if name == "webfaq-retrieval-english") == config.stage_1.epochs_webfaq


def test_stage_2_samples_webfaq_once(config, cache):
    source = cache["webfaq-retrieval-english"]
    sample = stage_records(source, config.stage_2, cache, config.resources.seed)
    assert len(sample) == 40
    assert np.array_equal(sample, stage_records(source, config.stage_2, cache, config.resources.seed))
    assert len(stage_records(source, config.stage_1, cache, config.resources.seed)) == len(source.train_records)


def test_source_records_and_source_epochs(workspace, tokenizer):
    config = make_config(workspace, stage_1=dict(source_records={"hotpotqa": 16}, source_epochs={"squad": 1, "hotpotqa": 3}))
    cache = open_cache(config, tokenizer)
    capped = stage_records(cache["hotpotqa"], config.stage_1, cache, config.resources.seed)
    assert len(capped) == 16 and set(capped.tolist()) <= set(cache["hotpotqa"].train_records.tolist())
    assert np.array_equal(capped, stage_records(cache["hotpotqa"], config.stage_1, cache, config.resources.seed))
    assert len(stage_records(cache["squad"], config.stage_1, cache, config.resources.seed)) == len(cache["squad"].train_records)
    plan = build_plan(cache, config.stage_1, 1, config.resources.seed)
    epochs = {}
    for index in range(plan.steps):
        name, epoch, records = plan.step(index)
        epochs.setdefault(name, set()).add(epoch)
        if name == "hotpotqa":
            assert set(records.tolist()) <= set(capped.tolist())
    assert epochs["squad"] == {0} and epochs["hotpotqa"] == {0, 1, 2}
    assert len(epochs["webfaq-retrieval-english"]) == config.stage_1.epochs_webfaq


def test_empty_source_settings_keep_the_plan(workspace, tokenizer):
    plain = make_config(workspace)
    explicit = make_config(workspace, stage_1=dict(source_records={}, source_epochs={}))
    cache = open_cache(plain, tokenizer)
    assert build_plan(cache, plain.stage_1, 1, 7).digest == build_plan(cache, explicit.stage_1, 1, 7).digest


def test_positive_choice_rotates():
    records = np.array([0, 1, 2])
    counts = np.array([2, 2, 1])
    first = positive_choice(records, counts, 0, seed=5)
    second = positive_choice(records, counts, 1, seed=5)
    assert (first[:2] != second[:2]).all() and first[2] == second[2] == 0


def test_step_layout_targets_and_duplicate_mask(config, cache):
    source = cache["squad"]
    records = source.train_records[:8]
    batch = build_step(source, records, epoch=0, seed=7, hard_negatives=0)
    assert batch.queries == 8 and batch.candidates == 8
    assert np.array_equal(batch.target, np.arange(8))
    for position, record in enumerate(records):
        assert np.array_equal(batch.tokens[position], source.query(int(record)))
    assert not batch.mask.any()  # all texts differ

    # The same record twice: each copy's positive is a false negative for the other.
    twice = np.array([records[0], records[1], records[0]])
    batch = build_step(source, twice, epoch=0, seed=7, hard_negatives=0)
    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 2] = expected[2, 0] = True
    assert np.array_equal(batch.mask, expected)


def test_other_positive_of_the_same_record_is_masked(config, cache):
    source = cache["hotpotqa"]  # 2 positives, 3 negatives per record
    records = source.train_records[:4]
    positives, _ = step_passages(source, records, 0, 7, 0)
    batch = build_step(source, records, 0, 7, 0)
    # Put record 0's other positive in as if it were a candidate of another query.
    other = source.record_start[records[0]] + (1 - (positives[0] - source.record_start[records[0]]))
    assert source.passage_hash[other] != source.passage_hash[positives[0]]
    assert not batch.mask.any()


def test_hard_negatives_own_first_then_mined(config, cache):
    source = cache["hotpotqa"]
    records = source.train_records[:4]
    positives, negatives = step_passages(source, records, 0, 7, hard_negatives=2)
    assert len(negatives) == 8
    for position, record in enumerate(records):
        own = set(range(source.record_start[record] + 2, source.record_start[record + 1]))
        assert set(negatives[2 * position : 2 * position + 2].tolist()) <= own
    batch = build_step(source, records, 0, 7, hard_negatives=2)
    assert batch.candidates == 4 + 8 and batch.mask.shape == (4, 12)

    plain = cache["squad"]  # no negatives of its own
    records = plain.train_records[:4]
    mined = np.full((plain.records, 2), -1, dtype=np.int32)
    mined[records[0]] = [5, 9]
    mined[records[2]] = [11, -1]
    _, negatives = step_passages(plain, records, 0, 7, 2, mined)
    assert negatives.tolist() == [5, 9, 11]
    _, negatives = step_passages(plain, records, 0, 7, 2, None)
    assert len(negatives) == 0


def test_chunks_respect_the_budget_and_cover_every_text():
    rng = np.random.default_rng(1)
    tokens = [rng.integers(3, 200, size=int(n)) for n in rng.integers(1, 40, size=57)]
    lengths = np.array([len(t) for t in tokens])
    chunks = make_chunks(tokens, budget=200)
    assert sorted(int(i) for chunk in chunks for i in chunk.index) == list(range(57))
    assert all(chunk.padded_tokens <= 200 for chunk in chunks)
    assert sum(chunk.padded_tokens for chunk in chunks) == padded_tokens(lengths, 200)
    assert chunks[0].input_ids.shape[1] == lengths.max()  # the longest texts come first
    for chunk in chunks:
        for row, position in enumerate(chunk.index.tolist()):
            length = lengths[position]
            assert np.array_equal(chunk.input_ids[row, :length].numpy(), tokens[position])
            assert chunk.attention_mask[row].sum() == length
            assert (chunk.input_ids[row, length:] == 0).all()
    # A text longer than the budget still gets a chunk of its own.
    assert [len(group) for group in chunk_order(np.array([500, 3]), 100)] == [1, 1]
