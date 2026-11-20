"""Training end to end on the tiny base: steps, resume, mining, stage 2."""

import json

import numpy as np
import pytest
import torch

from conftest import cpu_plan, make_config
from src.adapter_io import load_adapter
from src.base_model import full_base_hash, load_base
from src.errors import DataError, EmbeddingAdapterError
from src.estimate import stage_workload
from src.mining import load_mined, mine
from src.trainer import Trainer

QUIET = lambda _: None  # noqa: E731


def read_log(run_dir, name="train_log.jsonl"):
    return [json.loads(line) for line in (run_dir / name).open()]


@pytest.fixture(scope="module")
def stage_1_run(workspace, tmp_path_factory):
    """A complete stage-1 run with evaluations."""
    config = make_config(workspace)
    run_dir = tmp_path_factory.mktemp("stage_1")
    trainer = Trainer(config, 1, run_dir, cpu_plan(config), log=QUIET)
    before = full_base_hash(trainer.base.encoder)
    summary = trainer.train()
    return config, run_dir, trainer, before, summary


def test_training_runs_the_whole_plan_with_a_finite_falling_loss(stage_1_run):
    config, run_dir, trainer, _, summary = stage_1_run
    log = read_log(run_dir)
    assert summary["step"] == trainer.total_steps == len(log) == 26
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"]) for row in log)
    assert np.mean([row["loss"] for row in log[-5:]]) < np.mean([row["loss"] for row in log[:5]])
    evaluations = read_log(run_dir, "eval_log.jsonl")
    assert [row["step"] for row in evaluations] == [0, 4, 8, 12, 16, 20, 24, 26]
    assert evaluations[-1]["dev_ndcg10"] > evaluations[0]["dev_ndcg10"]
    assert summary["best_dev"] == max(row["dev_ndcg10"] for row in evaluations)
    for folder in ("final", "best"):
        assert (run_dir / folder / "adapter.safetensors").is_file()
    kept = sorted(path.name for path in run_dir.glob("checkpoint-*"))
    assert kept == ["checkpoint-0000024", "checkpoint-0000026"]  # keep_last = 2


def test_the_base_is_bit_identical_after_training(stage_1_run):
    config, _, trainer, before, _ = stage_1_run
    assert full_base_hash(trainer.base.encoder) == before
    assert before == full_base_hash(load_base(config.base_checkpoint).encoder)
    assert not any(p.requires_grad for n, p in trainer.base.encoder.named_parameters() if "lora_" not in n)


def test_the_log_counts_the_padded_tokens_the_estimator_counts(stage_1_run):
    config, run_dir, trainer, _, _ = stage_1_run
    workload = stage_workload(trainer.cache, config, 1, trainer.device.chunk_tokens)
    for name, entry in workload.items():
        rows = [row for row in read_log(run_dir) if row["source"] == name]
        assert entry["steps"] == len(rows)
        assert entry["padded_tokens"] == sum(row["padded_tokens"] for row in rows)
        assert entry["texts"] == sum(row["texts"] for row in rows)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_resume_gives_the_tensors_of_an_uninterrupted_run(workspace, tmp_path, dropout):
    config = make_config(workspace, adapter={"dropout": dropout})
    whole = Trainer(config, 1, tmp_path / "whole", cpu_plan(config), log=QUIET)
    whole.train(max_steps=10, evaluate=False)

    first = Trainer(config, 1, tmp_path / "parts", cpu_plan(config), log=QUIET)
    first.train(max_steps=5, evaluate=False)
    del first
    second = Trainer(config, 1, tmp_path / "parts", cpu_plan(config), log=QUIET)
    second.train(resume=True, max_steps=5, evaluate=False)
    assert second.step == whole.step == 10

    expected, got = whole.adapter.state_dict(), second.adapter.state_dict()
    assert all(torch.equal(expected[name], got[name]) for name in expected)
    a, b = whole.optimizer.state_dict()["state"], second.optimizer.state_dict()["state"]
    assert all(torch.equal(a[k]["exp_avg_sq"], b[k]["exp_avg_sq"]) for k in a)
    losses = [row["loss"] for row in read_log(tmp_path / "parts")]
    assert losses == [row["loss"] for row in read_log(tmp_path / "whole")]


def test_a_run_folder_is_not_overwritten_and_settings_may_not_change(workspace, tmp_path):
    config = make_config(workspace)
    Trainer(config, 1, tmp_path / "run", cpu_plan(config), log=QUIET).train(max_steps=2, evaluate=False)
    with pytest.raises(EmbeddingAdapterError, match="already holds checkpoints"):
        Trainer(config, 1, tmp_path / "run", cpu_plan(config), log=QUIET).train(max_steps=2, evaluate=False)
    changed = make_config(workspace, stage_1={"lr": 1e-3})
    with pytest.raises(EmbeddingAdapterError, match="config_digest"):
        Trainer(changed, 1, tmp_path / "run", cpu_plan(changed), log=QUIET).train(resume=True, evaluate=False)
    with pytest.raises(EmbeddingAdapterError, match="nothing to resume"):
        Trainer(config, 1, tmp_path / "empty", cpu_plan(config), log=QUIET).train(resume=True, evaluate=False)


def test_mining_then_stage_2(stage_1_run, tmp_path):
    config, run_dir, trainer, before, _ = stage_1_run
    cache = trainer.cache
    base = load_base(config.base_checkpoint)
    load_adapter(run_dir / "final", base)
    document = mine(base.encoder, cache, config, cpu_plan(config), tmp_path / "mined", {"path": "final"}, log=QUIET)
    tables = load_mined(tmp_path / "mined", cache)
    assert set(tables) == set(cache.names) == set(document["sources"])
    for name, table in tables.items():
        source = cache[name]
        assert table.shape == (source.records, config.mining.per_query)
        assert (table[source.dev_records] == -1).all()  # dev queries are never mined
        assert (table >= 0).any()
        for record in np.flatnonzero((table >= 0).any(axis=1)):
            own = set(source.passage_hash[source.record_start[record] : source.record_start[record + 1]].tolist())
            found = table[record][table[record] >= 0]
            assert len(set(found.tolist())) == len(found)
            assert not own & set(source.passage_hash[found].tolist())
    assert len(np.flatnonzero((tables["webfaq-retrieval-english"] >= 0).any(axis=1))) <= config.stage_2.webfaq_sample

    stage_2 = Trainer(config, 2, tmp_path / "stage_2", cpu_plan(config), init_adapter=run_dir / "final",
                      mined_dir=tmp_path / "mined", log=QUIET)  # fmt: skip
    started = stage_2.adapter.state_dict()
    assert all(torch.equal(started[n], t) for n, t in trainer.adapter.state_dict().items())
    summary = stage_2.train()
    log = read_log(tmp_path / "stage_2")
    assert summary["step"] == stage_2.total_steps == len(log)
    assert all(np.isfinite(row["loss"]) for row in log)
    assert max(row["texts"] for row in log) > 2 * config.stage_2.batch_queries  # hard negatives were added
    assert full_base_hash(stage_2.base.encoder) == before
    # With the tables, the estimator counts exactly what the run encoded.
    workload = stage_workload(cache, config, 2, stage_2.device.chunk_tokens, mined=tables)
    assert sum(e["padded_tokens"] for e in workload.values()) == sum(row["padded_tokens"] for row in log)


def test_mined_tables_of_another_cache_are_refused(stage_1_run, tmp_path):
    config, _, trainer, _, _ = stage_1_run
    (tmp_path / "mined").mkdir()
    (tmp_path / "mined" / "mined.json").write_text(json.dumps({"cache_digest": "other", "sources": {}}))
    with pytest.raises(DataError, match="different token cache"):
        load_mined(tmp_path / "mined", trainer.cache)
    with pytest.raises(DataError, match="embed.py mine"):
        load_mined(tmp_path / "nothing", trainer.cache)
