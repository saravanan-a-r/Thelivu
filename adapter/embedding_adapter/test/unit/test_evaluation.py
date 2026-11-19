"""Metrics, the benchmark reader and small helpers."""

import math
import pickle

import numpy as np
import pytest
import torch

from conftest import cpu_plan, make_config
from src.base_model import load_base
from src.estimate import evaluation_counts
from src.evaluation import Benchmark, build_dev_tasks, evaluate_dev, ndcg_at_10, retrieval_average
from src.resources import parse_gpu_processes
from src.text import TextTokenizer
from src.trainer import learning_rate


def test_ndcg_matches_a_hand_computed_example():
    documents = torch.eye(4)
    # Query 0 ranks the documents 0, 1, 2, 3; query 1 ranks them 3, 2, 1, 0.
    queries = torch.nn.functional.normalize(torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]), dim=1)
    gains = [{1: 1.0}, {3: 2.0, 0: 1.0}]
    ndcg, recall = ndcg_at_10(queries, documents, gains)
    first = (1 / math.log2(3)) / 1.0
    second = (2 / math.log2(2) + 1 / math.log2(5)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg == pytest.approx((first + second) / 2, rel=1e-9)
    assert recall == 1.0

    # Excluding document 0 for query 0 moves its relevant document to rank 1.
    ndcg, _ = ndcg_at_10(queries[:1], documents, gains[:1], exclude=[0])
    assert ndcg == pytest.approx(1.0)
    # A judged document that is not in the corpus lowers the score, as it can never be found.
    ndcg, recall = ndcg_at_10(queries[:1], documents, [{0: 1.0, -1: 1.0}])
    assert ndcg == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert recall == 0.5


def test_benchmark_reads_a_task_and_scores_it(workspace, tmp_path):
    tasks = tmp_path / "eval" / "tasks"
    tasks.mkdir(parents=True)
    retrieval = {
        "qids": ["q1", "q2"],
        "q": ["what is the river of paris alpha one?", "what is the lake of rome beta two?"],
        "dids": ["d1", "d2", "d3"],
        "d": ["the river in paris is named alpha one", "the lake in rome is named beta two", "the fox lives"],
        "rel": {"q1": {"d1": 1, "gone": 1}, "q2": {"d2": 2}},
    }
    with (tasks / "scifact.pkl").open("wb") as handle:
        pickle.dump(retrieval, handle)
    symmetric = {
        "qids": ["d1", "d2"],
        "q": retrieval["d"][:2],
        "dids": ["d1", "d2", "d3"],
        "d": retrieval["d"][:2] + ["the river in paris is called alpha one"],
        "rel": {"d1": {"d1": 1, "d3": 1}, "d2": {"d2": 1}},
    }
    with (tasks / "arguana.pkl").open("wb") as handle:
        pickle.dump(symmetric, handle)
    with (tasks / "sts.pkl").open("wb") as handle:
        pickle.dump({"a": ["the red river", "the fox", "one two"], "b": ["the blue river", "ten lima", "one two"],
                     "gold": [3.0, 0.0, 5.0]}, handle)  # fmt: skip
    config = make_config(workspace)
    config = type(config)(**{**config.__dict__, "eval_root": tmp_path / "eval"})
    benchmark = Benchmark(config, TextTokenizer.from_checkpoint(config.base_checkpoint))
    task = benchmark._task("scifact")
    assert task["gains"][0] == {0: 1.0, -1: 1.0} and task["exclude"] is None
    assert benchmark._task("arguana")["exclude"] == [0, 1]
    assert benchmark._task("arguana")["gains"][0] == {2: 1.0}  # the query itself is not a target
    assert benchmark.tokens("sts") == int(benchmark.lengths("sts").sum())

    encoder = load_base(config.base_checkpoint).encoder
    result = benchmark.evaluate(encoder, ["scifact", "arguana", "sts"], cpu_plan(config), dims=(None, 16))
    assert set(result) >= {"bench/scifact", "bench/scifact@16", "recall100/scifact", "bench/arguana", "bench/sts"}
    assert all(-1.0 <= value <= 1.0 for value in result.values())
    assert retrieval_average(result) is None


def test_dev_tasks_are_fixed_and_hold_every_gold_passage(config, workspace):
    from src.cache import open_cache

    cache = open_cache(config, TextTokenizer.from_checkpoint(config.base_checkpoint))
    tasks = build_dev_tasks(cache, config)
    again = build_dev_tasks(cache, config)
    assert [task.name for task in tasks] == cache.names
    for task, other in zip(tasks, again):
        assert len(task.queries) == len(cache[task.name].dev_records)
        assert all(np.array_equal(a, b) for a, b in zip(task.documents, other.documents))
        assert all(gain and all(0 <= d < len(task.documents) for d in gain) for gain in task.gains)
    result = evaluate_dev(load_base(config.base_checkpoint).encoder, tasks, cpu_plan(config))
    assert set(result) == {f"dev/{name}" for name in cache.names} | {"dev_ndcg10"}
    assert result["dev_ndcg10"] == pytest.approx(np.mean([result[f"dev/{name}"] for name in cache.names]))


def test_parse_gpu_processes_drops_itself_and_bad_lines():
    output = "664727, 33480\n 1234, 512\nNo running processes found\n99, [N/A]\n"
    assert parse_gpu_processes(output, own_pid=1234) == [(664727, 33480), (99, 0)]
    assert parse_gpu_processes("", own_pid=1) == []


def test_evaluation_counts():
    assert evaluation_counts(5000, 1000, 2000) == (6, 3)  # 0, 1000 ... 5000; benchmark at 0, 2000, 4000
    assert evaluation_counts(5500, 1000, 2000) == (7, 3)  # plus the final step
    assert evaluation_counts(100, 0, 2000) == (0, 0)
    assert evaluation_counts(4000, 1000, 0) == (5, 0)


def test_learning_rate_warms_up_then_decays_to_the_floor(config):
    stage = config.stage_1
    total = 20
    rates = [learning_rate(step, total, stage) for step in range(total)]
    assert rates[0] == pytest.approx(stage.lr / stage.warmup_steps)
    assert max(rates) == pytest.approx(stage.lr)
    assert all(a >= b for a, b in zip(rates[stage.warmup_steps :], rates[stage.warmup_steps + 1 :]))
    assert rates[-1] >= stage.lr * stage.min_lr_ratio - 1e-12
