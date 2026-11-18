"""
How long a full training run takes on this machine, from measurements.

Two things are measured on the real base model with real data:

  * seconds per padded token of a training step (both GradCache passes, the
    loss and the optimizer update), separately for every source and stage;
  * seconds per padded token of encoding without gradients (evaluations and
    mining).

Everything else is counted, not measured: the step plan gives every step's
records, the token cache their lengths, and `padded_tokens` the exact number
of padded tokens the chunker will hand to the encoder. The duration of a
stage is the sum over its sources of (padded tokens x measured seconds per
padded token), plus its evaluations counted the same way.

One number is not known before stage 1 has run: which hard negatives mining
will pick for stage 2. Their count is taken as the maximum (every query gets
`hard_negatives`) and their lengths are drawn from the passages of the same
source, which is the pool mining draws from.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .batching import build_step, make_chunks, padded_tokens, step_passages
from .cache import TokenCache
from .config import EmbeddingConfig
from .encoding import encode_tokens
from .evaluation import ALL_TASKS, Benchmark, build_dev_tasks
from .plan import StepPlan, build_plan, stage_records
from .resources import DevicePlan, peak_memory_gb, reset_peak_memory
from .trainer import PreparedStep, Trainer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


def stage_workload(
    cache: TokenCache,
    config: EmbeddingConfig,
    stage_number: int,
    budget: int,
    *,
    steps: int | None = None,
    mined: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, int]]:
    """Per source: steps, texts and padded tokens of the stage's first `steps` steps (all by default)."""
    stage = config.stage(stage_number)
    seed = config.resources.seed
    plan = build_plan(cache, stage, stage_number, seed)
    rng = np.random.default_rng([seed, 17, stage_number])
    workload: dict[str, dict[str, int]] = {}
    for index in range(plan.steps if steps is None else min(steps, plan.steps)):
        name, epoch, records = plan.step(index)
        source = cache[name]
        table = None if mined is None else mined.get(name)
        positives, negatives = step_passages(source, records, epoch, seed, stage.hard_negatives, table)
        lengths = [source.query_lengths[records], source.passage_lengths[positives], source.passage_lengths[negatives]]
        if stage.hard_negatives > 0 and table is None:
            missing = stage.hard_negatives * len(records) - len(negatives)
            lengths.append(source.passage_lengths[rng.integers(0, source.passages, size=missing)])
        lengths = np.concatenate(lengths).astype(np.int64)
        entry = workload.setdefault(name, {"steps": 0, "texts": 0, "padded_tokens": 0, "tokens": 0})
        entry["steps"] += 1
        entry["texts"] += len(lengths)
        entry["tokens"] += int(lengths.sum())
        entry["padded_tokens"] += padded_tokens(lengths, budget)
    return workload


def evaluation_counts(total_steps: int, eval_every: int, benchmark_every: int) -> tuple[int, int]:
    """(dev evaluations, benchmark evaluations) the training loop runs in `total_steps` steps."""
    if eval_every <= 0:
        return 0, 0
    at = set(range(0, total_steps + 1, eval_every)) | {total_steps}
    benchmarks = sum(1 for step in at if benchmark_every > 0 and step % benchmark_every == 0)
    return len(at), benchmarks


def dev_padded_tokens(cache: TokenCache, config: EmbeddingConfig, budget: int) -> int:
    total = 0
    for task in build_dev_tasks(cache, config):
        lengths = np.asarray([len(ids) for ids in task.queries + task.documents], dtype=np.int64)
        total += padded_tokens(lengths, budget)
    return total


def benchmark_padded_tokens(benchmark: Benchmark, names: Sequence[str], budget: int) -> int:
    return sum(padded_tokens(benchmark.lengths(name), budget) for name in names)


def mining_padded_tokens(cache: TokenCache, config: EmbeddingConfig, budget: int) -> int:
    """Padded tokens mining encodes: each source's distinct passages and its queries."""
    total = 0
    for name in cache.names:
        source = cache[name]
        records = stage_records(source, config.stage_2, cache, config.resources.seed)
        spans = [np.arange(source.record_start[r], source.record_start[r + 1]) for r in records]
        passages = np.concatenate(spans)
        _, first = np.unique(source.passage_hash[passages], return_index=True)
        total += padded_tokens(source.passage_lengths[passages[first]].astype(np.int64), budget)
        total += padded_tokens(source.query_lengths[records].astype(np.int64), budget)
    return total


# --------------------------------------------------------------------------
# measuring
# --------------------------------------------------------------------------


def first_steps(plan: StepPlan, per_source: int) -> dict[str, list[int]]:
    """The first `per_source` steps of every source in the plan."""
    found: dict[str, list[int]] = {}
    for index in range(plan.steps):
        name = plan.sources[int(plan.source_of_step[index])]
        steps = found.setdefault(name, [])
        if len(steps) < per_source:
            steps.append(index)
    return found


def prepare_sample(trainer: Trainer, index: int, sample_queries: int) -> PreparedStep:
    """Step `index` of the plan, or its first `sample_queries` queries (0 = the whole step)."""
    if sample_queries <= 0:
        return trainer.prepare_step(index)
    name, epoch, records = trainer.plan.step(index)
    batch = build_step(
        trainer.cache[name], records[:sample_queries], epoch, trainer.seed,
        trainer.stage.hard_negatives, trainer.mined.get(name),
    )  # fmt: skip
    chunks = make_chunks(batch.tokens, trainer.device.chunk_tokens, trainer.tokenizer.pad_id)
    return PreparedStep(index, batch, chunks)


def measure_training(
    trainer: Trainer,
    *,
    steps_per_source: int,
    sample_queries: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, float]]:
    """
    Seconds per padded token of real training steps, per source.

    The adapter and optimizer of `trainer` are updated by the measured steps;
    use a trainer made for measuring.
    """
    device = trainer.device.device
    chosen = first_steps(trainer.plan, steps_per_source)
    warm = prepare_sample(trainer, next(iter(chosen.values()))[0], sample_queries)
    trainer.train_step(warm)  # first-call costs (allocator, kernels) are not part of a step
    synchronize(device)
    reset_peak_memory(device)
    result: dict[str, dict[str, float]] = {}
    for name, indices in chosen.items():
        seconds, tokens, loss = 0.0, 0.0, 0.0
        for index in indices:
            prepared = prepare_sample(trainer, index, sample_queries)
            synchronize(device)
            started = time.perf_counter()
            stats = trainer.train_step(prepared)
            synchronize(device)
            seconds += time.perf_counter() - started
            tokens += stats["padded_tokens"]
            loss = stats["loss"]
        result[name] = {
            "seconds_per_token": seconds / tokens,
            "tokens_per_second": tokens / seconds,
            "measured_tokens": tokens,
            "measured_seconds": seconds,
            "last_loss": loss,
        }
        log(f"  stage {trainer.stage_number} {name}: {tokens / seconds:,.0f} padded tok/s "
            f"({len(indices)} steps, {seconds:.1f} s)")
    result["_peak_memory_gb"] = {"value": peak_memory_gb(device)}
    return result


def measure_encoding(
    encoder: torch.nn.Module, tokens: list[np.ndarray], plan: DevicePlan, max_tokens: int
) -> dict[str, float]:
    """Seconds per padded token of encoding without gradients, on about `max_tokens` padded tokens."""
    lengths = np.asarray([len(ids) for ids in tokens], dtype=np.int64)
    keep = max(1, min(len(tokens), int(max_tokens / max(1.0, lengths.mean()))))
    sample = tokens[:keep]
    encode_tokens(encoder, sample[: max(1, keep // 8)], plan)  # warm-up
    synchronize(plan.device)
    started = time.perf_counter()
    encode_tokens(encoder, sample, plan)
    synchronize(plan.device)
    seconds = time.perf_counter() - started
    padded = padded_tokens(lengths[:keep], plan.encode_tokens)
    return {"seconds_per_token": seconds / padded, "tokens_per_second": padded / seconds,
            "measured_tokens": float(padded), "measured_seconds": seconds}  # fmt: skip


# --------------------------------------------------------------------------
# the estimate
# --------------------------------------------------------------------------

PILOT_STEPS = 1500
PILOT_RUNS = 3


def estimate(
    config: EmbeddingConfig,
    plan: DevicePlan,
    *,
    steps_per_source: int,
    sample_queries: int,
    encode_sample_tokens: int,
    scratch: str,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure, count, and return the duration of every part of the full run in seconds."""
    report: dict[str, Any] = {"device": plan.describe(), "chunk_tokens": plan.chunk_tokens,
                              "encode_tokens": plan.encode_tokens, "parts": {}}  # fmt: skip
    rates: dict[int, dict[str, dict[str, float]]] = {}
    trainer = None
    for stage_number in (1, 2):
        if trainer is not None:
            # One encoder in memory at a time.
            del trainer
            if plan.is_gpu:
                torch.cuda.empty_cache()
        trainer = Trainer(config, stage_number, f"{scratch}/stage_{stage_number}", plan, log=lambda _: None)
        log(f"measuring stage {stage_number} training steps")
        rates[stage_number] = measure_training(
            trainer, steps_per_source=steps_per_source, sample_queries=sample_queries, log=log
        )
    assert trainer is not None
    cache, encoder = trainer.cache, trainer.base.encoder
    report["peak_memory_gb"] = max(rates[s]["_peak_memory_gb"]["value"] for s in rates)

    log("measuring encoding without gradients")
    benchmark = Benchmark(config, trainer.tokenizer)
    dev_tasks = build_dev_tasks(cache, config)
    dev_rate = measure_encoding(encoder, dev_tasks[0].documents, plan, encode_sample_tokens)
    bench_rate = measure_encoding(encoder, benchmark._task("scifact")["d"], plan, encode_sample_tokens)
    log(f"  cache passages: {dev_rate['tokens_per_second']:,.0f} padded tok/s; "
        f"benchmark documents: {bench_rate['tokens_per_second']:,.0f} padded tok/s")
    report["rates"] = {"train": rates, "encode_cache": dev_rate, "encode_benchmark": bench_rate}

    log("counting the padded tokens of the full run")
    dev_tokens = dev_padded_tokens(cache, config, plan.encode_tokens)
    fast_tokens = benchmark_padded_tokens(benchmark, config.dev.benchmark_tasks, plan.encode_tokens)
    full_tokens = benchmark_padded_tokens(benchmark, ALL_TASKS, plan.encode_tokens)
    dev_seconds = dev_tokens * dev_rate["seconds_per_token"]
    fast_seconds = fast_tokens * bench_rate["seconds_per_token"]

    def stage_part(stage_number: int, steps: int | None = None) -> dict[str, Any]:
        stage = config.stage(stage_number)
        workload = stage_workload(cache, config, stage_number, plan.chunk_tokens, steps=steps)
        total_steps = sum(entry["steps"] for entry in workload.values())
        train = sum(
            entry["padded_tokens"] * rates[stage_number][name]["seconds_per_token"]
            for name, entry in workload.items()
        )
        dev_runs, bench_runs = evaluation_counts(total_steps, stage.eval_every, config.dev.benchmark_every)
        return {
            "steps": total_steps,
            "padded_tokens": sum(entry["padded_tokens"] for entry in workload.values()),
            "train_seconds": train,
            "dev_evaluations": dev_runs,
            "benchmark_evaluations": bench_runs,
            "evaluation_seconds": dev_runs * dev_seconds + bench_runs * fast_seconds,
            "seconds": train + dev_runs * dev_seconds + bench_runs * fast_seconds,
            "by_source": workload,
        }

    parts = report["parts"]
    pilot = stage_part(1, PILOT_STEPS)
    pilot = {**pilot, "runs": PILOT_RUNS, "seconds": pilot["seconds"] * PILOT_RUNS}
    pilot.pop("by_source")
    parts["rank_pilot"] = pilot
    parts["stage_1"] = stage_part(1)
    mining_tokens = mining_padded_tokens(cache, config, plan.encode_tokens)
    parts["mining"] = {"padded_tokens": mining_tokens, "seconds": mining_tokens * dev_rate["seconds_per_token"]}
    parts["stage_2"] = stage_part(2)
    # Full benchmark of stage-1 best (gate G3) and of stage-2 best (final report).
    parts["full_benchmarks"] = {
        "runs": 2, "padded_tokens": 2 * full_tokens,
        "seconds": 2 * full_tokens * bench_rate["seconds_per_token"],
    }  # fmt: skip
    report["dev_evaluation_seconds"] = dev_seconds
    report["fast_benchmark_seconds"] = fast_seconds
    report["total_seconds"] = sum(part["seconds"] for part in parts.values())
    report["total_seconds_without_pilot"] = report["total_seconds"] - parts["rank_pilot"]["seconds"]
    return report


def format_report(report: dict[str, Any]) -> str:
    def hours(seconds: float) -> str:
        return f"{seconds / 3600:9.2f} h"

    lines = [f"device: {report['device']}  (chunk {report['chunk_tokens']:,} tokens, "
             f"peak memory {report['peak_memory_gb']:.1f} GB)"]  # fmt: skip
    names = {"rank_pilot": "rank pilot (3 x 1,500 steps)", "stage_1": "stage 1", "mining": "mining",
             "stage_2": "stage 2", "full_benchmarks": "full benchmark x 2"}  # fmt: skip
    for key, title in names.items():
        part = report["parts"][key]
        detail = f"{part['steps']:,} steps" if "steps" in part else f"{part['padded_tokens']:,} tokens"
        lines.append(f"  {title:30s} {hours(part['seconds'])}   {detail}")
    lines.append(f"  {'total':30s} {hours(report['total_seconds'])}")
    lines.append(f"  {'total without the rank pilot':30s} {hours(report['total_seconds_without_pilot'])}")
    return "\n".join(lines)
