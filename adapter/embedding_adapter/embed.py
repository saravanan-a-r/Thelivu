#!/usr/bin/env python
"""
Embedding adapter for the Thelivu base model: one entry point.

    embed.py prepare                               tokenise the training data once (CPU)
    embed.py info                                  what the token cache and the step plans hold
    embed.py benchmark --device gpu                tokens/s and peak memory per chunk budget
    embed.py estimate  --device gpu|cpu            measured duration of the full run
    embed.py train --stage 1 --run NAME            stage 1: in-batch negatives
    embed.py mine  --run NAME                      hard negatives with the stage-1 adapter
    embed.py train --stage 2 --run NAME            stage 2: hard negatives
    embed.py evaluate --adapter DIR                benchmark an adapter (or the plain base)
    embed.py encode --adapter DIR --query "..."    vectors or late chunks for a text

A run lives in runs/NAME/: stage_1/, mined/, stage_2/. Every command takes
--config and --set section.key=value.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "embedding_config.yml"
RUNS = ROOT / "runs"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from src.errors import EmbeddingAdapterError

    try:
        return args.handler(args) or 0
    except EmbeddingAdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler, summary: str, device: bool = False) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.set_defaults(handler=handler)
        sub.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML settings file")
        sub.add_argument("--set", action="append", default=[], metavar="section.key=value",
                         help="override one setting (repeatable)")  # fmt: skip
        if device:
            sub.add_argument("--device", choices=("gpu", "cpu"), required=True)
            sub.add_argument("--gpu-memory-gb", type=float, default=None,
                             help="memory this process may use on the GPU (default: resources.gpu_memory_gb)")  # fmt: skip
            sub.add_argument("--allow-busy-gpu", action="store_true",
                             help="start although another process holds GPU memory (it must be paused)")  # fmt: skip
        return sub

    sub = command("prepare", cmd_prepare, "tokenise the training data into the token cache")
    sub.add_argument("--workers", type=int, default=0, help="processes (default: all cores)")

    command("info", cmd_info, "show the token cache and the step plans")

    sub = command("benchmark", cmd_benchmark, "tokens/s and peak memory of stage-1 steps per chunk budget", device=True)
    sub.add_argument("--budgets", default="8000,12000,16000,24000", help="chunk budgets in padded tokens")
    sub.add_argument("--steps", type=int, default=20)
    sub.add_argument("--queries", type=int, default=0, help="queries per step (default: the full step)")

    sub = command("estimate", cmd_estimate, "measure and compute the duration of the full run", device=True)
    sub.add_argument("--steps-per-source", type=int, default=2, help="measured training steps per source and stage")
    sub.add_argument("--queries", type=int, default=0,
                     help="queries per measured step (default: the full step; use a few dozen on the CPU)")  # fmt: skip
    sub.add_argument("--encode-tokens", type=int, default=2_000_000, help="padded tokens of the encoding measurement")
    sub.add_argument("--out", default=None, help="write the report as JSON here")

    sub = command("train", cmd_train, "train one stage", device=True)
    sub.add_argument("--stage", type=int, choices=(1, 2), required=True)
    sub.add_argument("--run", required=True, help="run name (runs/NAME) or a path")
    sub.add_argument("--resume", action="store_true", help="continue from the last checkpoint of the stage")
    sub.add_argument("--max-steps", type=int, default=None, help="stop after this many steps of this call")
    sub.add_argument("--init", default=None,
                     help="adapter to start from: stage 2 default is the run's stage_1/best; "
                          "stage 1 default is a new adapter (give one to continue training on new data)")  # fmt: skip
    sub.add_argument("--mined", default=None, help="stage 2: mined negatives (default: the run's mined/)")
    sub.add_argument("--no-eval", action="store_true", help="skip every evaluation (smoke runs)")

    sub = command("mine", cmd_mine, "mine hard negatives for stage 2", device=True)
    sub.add_argument("--run", required=True)
    sub.add_argument("--adapter", default=None, help="adapter to mine with (default: the run's stage_1/best)")

    sub = command("evaluate", cmd_evaluate, "benchmark an adapter, or the plain base without --adapter", device=True)
    sub.add_argument("--adapter", default=None)
    sub.add_argument("--tasks", default="all", help="'all', 'fast' (the config's benchmark_tasks) or a comma list")
    sub.add_argument("--dims", default="", help="extra Matryoshka dimensions, e.g. 512,256,128")
    sub.add_argument("--dev", action="store_true", help="also score the dev split of the training sources")
    sub.add_argument("--out", default=None, help="write the scores as JSON here")

    sub = command("encode", cmd_encode, "embed a text, or late-chunk a document")
    sub.add_argument("--adapter", default=None, help="adapter folder (default: the plain base encoder)")
    sub.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    sub.add_argument("--load-decoder", action="store_true",
                     help="load the whole base model, not only its encoder")  # fmt: skip
    what = sub.add_mutually_exclusive_group(required=True)
    what.add_argument("--query", action="append", help="a query (repeatable)")
    what.add_argument("--passage", action="append", help="a passage (repeatable)")
    what.add_argument("--chunk-file", help="a text file to late-chunk")
    sub.add_argument("--chunk-tokens", type=int, default=None, help="--chunk-file: target tokens per chunk")
    sub.add_argument("--window-tokens", type=int, default=None,
                     help="--chunk-file: tokens the encoder reads at once (default: adapter's)")  # fmt: skip
    sub.add_argument("--overlap-tokens", type=int, default=None,
                     help="--chunk-file: tokens shared by neighbouring windows (default: adapter's)")  # fmt: skip
    sub.add_argument("--dim", type=int, default=None, help="vector size, one of the adapter's trained sizes")
    return parser


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------


def read_config(args: argparse.Namespace):
    from src.config import load_config
    from src.errors import ConfigError

    overrides = {}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            raise ConfigError(f"--set needs section.key=value, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return load_config(args.config, overrides)


def device_plan(args: argparse.Namespace, config):
    from src.resources import plan_device

    return plan_device(
        args.device, config, gpu_memory_gb=args.gpu_memory_gb, allow_busy_gpu=args.allow_busy_gpu
    )


def run_dir(name: str) -> Path:
    return Path(name) if os.sep in name else RUNS / name


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    from src.cache import prepare_cache
    from src.resources import available_cores

    config = read_config(args)
    document = prepare_cache(config, args.workers or available_cores())
    records = sum(entry["records"] for entry in document["sources"].values())
    print(f"token cache ready at {config.cache_root}: {records:,} records, digest {document['digest']}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from src.cache import open_cache
    from src.plan import build_plan
    from src.text import TextTokenizer

    config = read_config(args)
    cache = open_cache(config, TextTokenizer.from_checkpoint(config.base_checkpoint))
    print(f"token cache {config.cache_root} (digest {cache.digest})")
    for name in cache.names:
        source = cache[name]
        print(f"  {name:28s} {source.records:>10,} records  {source.passages:>10,} passages  "
              f"dev {len(source.dev_records):>5,}  mean query {source.query_lengths.mean():5.1f} tok  "
              f"mean passage {source.passage_lengths.mean():6.1f} tok")  # fmt: skip
    for stage_number in (1, 2):
        plan = build_plan(cache, config.stage(stage_number), stage_number, config.resources.seed)
        print(f"stage {stage_number}: {plan.steps:,} steps of {config.stage(stage_number).batch_queries} queries "
              f"(plan {plan.digest})")  # fmt: skip
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    import torch

    from src.estimate import prepare_sample, synchronize
    from src.resources import peak_memory_gb, reset_peak_memory
    from src.trainer import Trainer

    config = read_config(args)
    plan = device_plan(args, config)
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for budget in (int(value) for value in args.budgets.split(",")):
            device = dataclasses.replace(plan, chunk_tokens=budget)
            trainer = Trainer(config, 1, scratch, device, log=lambda _: None)
            try:
                trainer.train_step(prepare_sample(trainer, 0, args.queries))  # warm-up
                synchronize(device.device)
                reset_peak_memory(device.device)
                seconds = tokens = 0.0
                for index in range(1, args.steps + 1):
                    prepared = prepare_sample(trainer, index, args.queries)
                    started = time.perf_counter()
                    stats = trainer.train_step(prepared)
                    synchronize(device.device)
                    seconds += time.perf_counter() - started
                    tokens += stats["padded_tokens"]
                row = {"chunk_tokens": budget, "tokens_per_second": tokens / seconds,
                       "seconds_per_step": seconds / args.steps, "peak_memory_gb": peak_memory_gb(device.device)}  # fmt: skip
                print(f"chunk {budget:>6,}: {row['tokens_per_second']:>9,.0f} padded tok/s  "
                      f"{row['seconds_per_step']:.2f} s/step  peak {row['peak_memory_gb']:.1f} GB")  # fmt: skip
            except torch.OutOfMemoryError:
                row = {"chunk_tokens": budget, "out_of_memory": True}
                print(f"chunk {budget:>6,}: out of memory")
            rows.append(row)
            del trainer
            if device.is_gpu:
                torch.cuda.empty_cache()
    print(json.dumps(rows))
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from src.estimate import estimate, format_report

    config = read_config(args)
    plan = device_plan(args, config)
    with tempfile.TemporaryDirectory() as scratch:
        report = estimate(
            config, plan,
            steps_per_source=args.steps_per_source, sample_queries=args.queries,
            encode_sample_tokens=args.encode_tokens, scratch=scratch,
        )  # fmt: skip
    print(format_report(report))
    if args.out:
        write_json(args.out, report)
        print(f"report written to {args.out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from src.errors import EmbeddingAdapterError
    from src.trainer import Trainer

    config = read_config(args)
    run = run_dir(args.run)
    init = mined = None
    if args.stage == 2:
        init = Path(args.init) if args.init else run / "stage_1" / "best"
        if not init.is_dir():
            raise EmbeddingAdapterError(f"stage 2 starts from a stage-1 adapter; {init} does not exist")
        mined = Path(args.mined) if args.mined else run / "mined"
        if not mined.is_dir():
            raise EmbeddingAdapterError(f"stage 2 needs mined negatives; {mined} does not exist (run `embed.py mine`)")
    else:
        if args.mined:
            raise EmbeddingAdapterError("--mined belongs to stage 2")
        if args.init:
            init = Path(args.init)
            if not init.is_dir():
                raise EmbeddingAdapterError(f"--init {init} does not exist")
    plan = device_plan(args, config)
    trainer = Trainer(config, args.stage, run / f"stage_{args.stage}", plan, init_adapter=init, mined_dir=mined)
    print(json.dumps(trainer.describe(), indent=2))
    result = trainer.train(resume=args.resume, max_steps=args.max_steps, evaluate=not args.no_eval)
    print(json.dumps(result))
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    from src.adapter_io import load_adapter
    from src.base_model import load_base
    from src.cache import open_cache
    from src.mining import mine
    from src.text import TextTokenizer

    config = read_config(args)
    run = run_dir(args.run)
    adapter_dir = Path(args.adapter) if args.adapter else run / "stage_1" / "best"
    plan = device_plan(args, config)
    cache = open_cache(config, TextTokenizer.from_checkpoint(config.base_checkpoint))
    base = load_base(config.base_checkpoint, load_decoder=False, device=plan.device)
    _, document = load_adapter(adapter_dir, base)
    info = {"path": str(adapter_dir), "extra": document["extra"]}
    mine(base.encoder, cache, config, plan, run / "mined", info)
    print(f"mined negatives written to {run / 'mined'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.adapter_io import load_adapter
    from src.base_model import load_base
    from src.cache import open_cache
    from src.evaluation import ALL_TASKS, Benchmark, build_dev_tasks, evaluate_dev, retrieval_average
    from src.text import TextTokenizer

    config = read_config(args)
    plan = device_plan(args, config)
    tokenizer = TextTokenizer.from_checkpoint(config.base_checkpoint)
    base = load_base(config.base_checkpoint, load_decoder=False, device=plan.device)
    if args.adapter:
        load_adapter(args.adapter, base)
    if args.tasks == "all":
        names = list(ALL_TASKS)
    elif args.tasks == "fast":
        names = list(config.dev.benchmark_tasks)
    else:
        names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    dims = [None] + [int(value) for value in args.dims.split(",") if value]
    result = Benchmark(config, tokenizer).evaluate(base.encoder, names, plan, dims)
    for dim in dims:
        average = retrieval_average(result, dim)
        if average is not None:
            result["retrieval_average" if dim is None else f"retrieval_average@{dim}"] = average
    if args.dev:
        cache = open_cache(config, tokenizer)
        result.update(evaluate_dev(base.encoder, build_dev_tasks(cache, config), plan))
    for key in sorted(result):
        print(f"  {key:32s} {result[key]:.4f}")
    if args.out:
        write_json(args.out, result)
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    from src.embedder import TextEmbedder
    from src.errors import EmbeddingAdapterError

    if not args.chunk_file and (args.window_tokens is not None or args.overlap_tokens is not None):
        raise EmbeddingAdapterError("--window-tokens and --overlap-tokens are only used with --chunk-file")
    config = read_config(args)
    embedder = TextEmbedder.from_pretrained(
        config.base_checkpoint, args.adapter, load_decoder=args.load_decoder, device=args.device
    )
    loaded = "encoder + decoder" if embedder.base.has_decoder else "encoder only"
    print(f"loaded {loaded}: {embedder.base.parameter_count():,} base parameters", file=sys.stderr)
    if args.chunk_file:
        text = Path(args.chunk_file).read_text(encoding="utf-8")
        chunks = embedder.late_chunk(
            text, chunk_tokens=args.chunk_tokens, dim=args.dim,
            window_tokens=args.window_tokens, overlap_tokens=args.overlap_tokens,
        )  # fmt: skip
        output = [{"start": c.start, "end": c.end, "tokens": c.tokens, "text": c.text,
                   "vector": c.vector.tolist()} for c in chunks]  # fmt: skip
    elif args.query:
        vectors = embedder.encode_queries(args.query, dim=args.dim)
        output = [{"text": text, "vector": vector.tolist()} for text, vector in zip(args.query, vectors)]
    else:
        vectors = embedder.encode_passages(args.passage, dim=args.dim)
        output = [{"text": text, "vector": vector.tolist()} for text, vector in zip(args.passage, vectors)]
    json.dump(output, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
