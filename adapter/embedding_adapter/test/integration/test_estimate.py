"""The time estimate: measured rates times counted tokens."""

import dataclasses
import pickle

import pytest

from conftest import cpu_plan, make_config
from src.estimate import PILOT_RUNS, estimate, format_report
from src.evaluation import ALL_TASKS, STS_TASK


def write_benchmark(root):
    tasks = root / "tasks"
    tasks.mkdir(parents=True)
    documents = [f"the river in paris is named alpha {word}." for word in "one two three four five six".split()]
    for name in ALL_TASKS:
        if name == STS_TASK:
            task = {"a": documents[:3], "b": documents[3:], "gold": [1.0, 2.0, 3.0]}
        else:
            task = {"qids": ["q1"], "q": ["what is the river of paris?"], "dids": [f"d{i}" for i in range(6)],
                    "d": documents, "rel": {"q1": {"d0": 1}}}  # fmt: skip
        with (tasks / f"{name}.pkl").open("wb") as handle:
            pickle.dump(task, handle)


def test_estimate_adds_up(workspace, tmp_path):
    write_benchmark(tmp_path / "eval")
    config = make_config(workspace, dev={"benchmark_tasks": ["scifact", "sts"], "benchmark_every": 8})
    config = dataclasses.replace(config, eval_root=tmp_path / "eval")
    report = estimate(config, cpu_plan(config), steps_per_source=2, sample_queries=0,
                      encode_sample_tokens=2000, scratch=str(tmp_path / "scratch"), log=lambda _: None)  # fmt: skip
    parts = report["parts"]
    assert set(parts) == {"rank_pilot", "stage_1", "mining", "stage_2", "full_benchmarks"}
    assert all(part["seconds"] > 0 for part in parts.values())
    assert report["total_seconds"] == pytest.approx(sum(part["seconds"] for part in parts.values()))
    assert report["total_seconds_without_pilot"] == pytest.approx(
        report["total_seconds"] - parts["rank_pilot"]["seconds"]
    )

    stage = parts["stage_1"]
    assert stage["steps"] == 26
    assert (stage["dev_evaluations"], stage["benchmark_evaluations"]) == (8, 4)  # 0, 4 ... 24, 26; 0, 8, 16, 24
    rates = report["rates"]["train"][1]
    by_hand = sum(entry["padded_tokens"] * rates[name]["seconds_per_token"] for name, entry in stage["by_source"].items())
    assert stage["train_seconds"] == pytest.approx(by_hand)
    assert stage["seconds"] == pytest.approx(
        by_hand + 8 * report["dev_evaluation_seconds"] + 4 * report["fast_benchmark_seconds"]
    )
    # The tiny plan is shorter than the pilot, so a pilot run is one whole stage 1.
    assert parts["rank_pilot"]["seconds"] == pytest.approx(PILOT_RUNS * stage["seconds"])
    assert not (tmp_path / "scratch").exists()  # measuring writes nothing
    assert "total without the rank pilot" in format_report(report)
