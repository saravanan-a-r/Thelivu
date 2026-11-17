"""
The training loop for one stage.

What a run directory holds:

    run_config.json            the settings the run was started with
    train_log.jsonl            one line per step
    eval_log.jsonl             one line per evaluation
    checkpoint-0000500/        adapter + optimizer state, resumable
    best/                      the adapter with the highest dev nDCG@10
    final/                     the adapter after the last step

Resume is exact: the step plan is a function of the seed and the data, the
learning rate a function of the step, the dropout masks a function of
(seed, step, chunk). A checkpoint therefore needs only the adapter, the
optimizer and the step number, and a run stopped at any checkpoint continues
with the tensors an uninterrupted run would have had.

Only the encoder is loaded. Stage 1 starts from a fresh adapter (or, to continue
training on new data, from `init_adapter`); stage 2
starts from a stage-1 adapter (`init_adapter`) with a new optimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from safetensors.torch import load_file

from .adapter_io import ADAPTER_WEIGHTS, embedding_contract, load_adapter, save_adapter
from .base_model import BaseModel, load_base
from .batching import Chunk, StepBatch, build_step, make_chunks
from .cache import TokenCache, open_cache
from .config import EmbeddingConfig, StageConfig
from .errors import EmbeddingAdapterError
from .evaluation import Benchmark, build_dev_tasks, evaluate_dev
from .gradcache import gradcache_step
from .lora import EncoderAdapter, attach_adapter
from .loss import contrastive_loss
from .mining import load_mined
from .plan import StepPlan, build_plan
from .pooling import pooled
from .resources import DevicePlan, peak_memory_gb
from .text import TextTokenizer

STATE_FILE = "trainer_state.pt"
CHECKPOINT_PREFIX = "checkpoint-"


def learning_rate(step: int, total: int, stage: StageConfig) -> float:
    """Linear warm-up, then cosine down to `min_lr_ratio * lr` at the last step."""
    if step < stage.warmup_steps:
        return stage.lr * (step + 1) / stage.warmup_steps
    span = max(1, total - stage.warmup_steps)
    progress = min(1.0, (step - stage.warmup_steps) / span)
    floor = stage.min_lr_ratio
    return stage.lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def step_seed(seed: int, stage: int, step: int) -> int:
    """Seed of a step's dropout masks; chunks add their index to it."""
    return (seed * 1_000_003 + stage * 7_919 + step) * 4_096 % (2**62)


@dataclass
class PreparedStep:
    index: int
    batch: StepBatch
    chunks: list[Chunk]


class StepFeeder:
    """Builds the coming steps on a helper thread so the device never waits for data."""

    def __init__(self, make: Callable[[int], PreparedStep], start: int, stop: int, depth: int = 6) -> None:
        self._make = make
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(start, stop), daemon=True)
        self._thread.start()

    def _run(self, start: int, stop: int) -> None:
        try:
            for index in range(start, stop):
                item: Any = self._make(index)
                while not self._closed.is_set():
                    try:
                        self._queue.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self._closed.is_set():
                    return
        except BaseException as exc:  # handed to the training thread
            self._queue.put(exc)

    def next(self) -> PreparedStep:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._closed.set()
        while self._thread.is_alive():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                self._thread.join(timeout=0.2)


class Trainer:
    def __init__(
        self,
        config: EmbeddingConfig,
        stage_number: int,
        run_dir: str | Path,
        device: DevicePlan,
        *,
        init_adapter: str | Path | None = None,
        mined_dir: str | Path | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.stage_number = stage_number
        self.stage = config.stage(stage_number)
        self.run_dir = Path(run_dir)
        self.device = device
        self.log = log
        self.seed = config.resources.seed

        self.tokenizer = TextTokenizer.from_checkpoint(config.base_checkpoint)
        self.cache: TokenCache = open_cache(config, self.tokenizer)
        self.plan: StepPlan = build_plan(self.cache, self.stage, stage_number, self.seed)
        self.mined = self._load_mined(mined_dir)

        self.base: BaseModel = load_base(config.base_checkpoint, load_decoder=False, device=device.device)
        self._check_dims()
        self.init_adapter = None if init_adapter is None else str(init_adapter)
        if init_adapter is None:
            torch.manual_seed(self.seed)  # the LoRA initialisation is part of the run
            self.adapter: EncoderAdapter = attach_adapter(self.base.encoder, config.adapter)
        else:
            self.adapter, document = load_adapter(init_adapter, self.base)
            if document["adapter"]["rank"] != config.adapter.rank:
                raise EmbeddingAdapterError(
                    f"{init_adapter} has rank {document['adapter']['rank']}, "
                    f"the config asks for {config.adapter.rank}"
                )
            self._adopt_dropout()
        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=self.stage.lr,
            betas=tuple(config.optim.betas),
            weight_decay=config.optim.weight_decay,
            fused=device.is_gpu,
        )
        self.step = 0
        self.best_dev: float | None = None
        self._stop_requested = False
        self._dev_tasks = None
        self._benchmark = None

    # -- setup -------------------------------------------------------------

    def _check_dims(self) -> None:
        d_model = self.base.config.d_model
        too_large = [d for d in self.config.loss.matryoshka_dims if d > d_model]
        if too_large:
            raise EmbeddingAdapterError(f"matryoshka_dims {too_large} exceed d_model {d_model}")

    def _adopt_dropout(self) -> None:
        """An adapter loaded for further training trains with this config's dropout."""
        rate = self.config.adapter.dropout
        for module in self.adapter.modules.values():
            module.dropout = torch.nn.Dropout(rate) if rate > 0 else torch.nn.Identity()

    def _load_mined(self, mined_dir: str | Path | None) -> dict[str, np.ndarray]:
        if self.stage.hard_negatives == 0:
            return {}
        if mined_dir is None:
            self.log("no mined negatives given: only the records' own negatives are used")
            return {}
        return load_mined(mined_dir, self.cache)

    @property
    def total_steps(self) -> int:
        return self.plan.steps

    def describe(self) -> dict[str, Any]:
        return {
            "stage": self.stage_number,
            "steps": self.total_steps,
            "batch_queries": self.stage.batch_queries,
            "hard_negatives": self.stage.hard_negatives,
            "adapter_parameters": self.adapter.parameter_count(),
            "base_parameters_loaded": self.base.parameter_count(),
            "base_fingerprint": self.base.fingerprint,
            "plan_digest": self.plan.digest,
            "cache_digest": self.cache.digest,
            "config_digest": self._config_digest(),
            "device": self.device.describe(),
            "chunk_tokens": self.device.chunk_tokens,
            "init_adapter": self.init_adapter,
        }

    def _config_digest(self) -> str:
        """Settings that decide the tensors of a run (not how often it saves or evaluates)."""
        stage = self.config.to_dict()[f"stage_{self.stage_number}"]
        for key in ("eval_every", "save_every", "keep_last"):
            stage.pop(key)
        payload = {
            "adapter": self.config.to_dict()["adapter"],
            "loss": self.config.to_dict()["loss"],
            "optim": self.config.to_dict()["optim"],
            "stage": stage,
            "seed": self.seed,
            "chunk_tokens": self.device.chunk_tokens,
            "bf16": self.device.bf16,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    # -- one step ----------------------------------------------------------

    def prepare_step(self, index: int) -> PreparedStep:
        name, epoch, records = self.plan.step(index)
        batch = build_step(
            self.cache[name], records, epoch, self.seed, self.stage.hard_negatives, self.mined.get(name)
        )
        chunks = make_chunks(batch.tokens, self.device.chunk_tokens, self.tokenizer.pad_id)
        if self.device.is_gpu:
            chunks = [chunk.pin() for chunk in chunks]
        return PreparedStep(index, batch, chunks)

    def train_step(self, prepared: PreparedStep) -> dict[str, float]:
        """Forward, backward and optimizer update for one prepared step."""
        device = self.device.device
        batch = prepared.batch
        encoder = self.base.encoder
        encoder.train()  # only the LoRA dropout reacts; the base has none
        chunks = [chunk.to(device) for chunk in prepared.chunks]
        target = torch.from_numpy(batch.target).to(device)
        mask = torch.from_numpy(batch.mask).to(device)
        queries = batch.queries
        loss_config = self.config.loss

        def encode(chunk: Chunk) -> torch.Tensor:
            return pooled(encoder, chunk.input_ids, chunk.attention_mask, self.device.bf16)

        def loss_fn(vectors: torch.Tensor):
            return contrastive_loss(
                vectors[:queries], vectors[queries:], target, mask,
                loss_config.temperature, loss_config.matryoshka_dims,
            )  # fmt: skip

        rate = learning_rate(prepared.index, self.total_steps, self.stage)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        loss, stats = gradcache_step(
            chunks, encode, loss_fn, step_seed(self.seed, self.stage_number, prepared.index)
        )
        if not math.isfinite(loss):
            raise EmbeddingAdapterError(f"the loss is {loss} at step {prepared.index}; stopping")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.config.optim.grad_clip)
        self.optimizer.step()
        return {
            "loss": loss,
            "lr": rate,
            "grad_norm": float(grad_norm),
            "padded_tokens": float(sum(chunk.padded_tokens for chunk in chunks)),
            "texts": float(len(batch.tokens)),
            **stats,
        }

    # -- the loop ----------------------------------------------------------

    def train(self, *, resume: bool = False, max_steps: int | None = None, evaluate: bool = True) -> dict[str, Any]:
        """
        Run to the end of the plan, or `max_steps` steps of this call.

        `evaluate=False` skips every evaluation (smoke runs and timing).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self._resume()
        elif any(self.run_dir.glob(f"{CHECKPOINT_PREFIX}*")):
            raise EmbeddingAdapterError(
                f"{self.run_dir} already holds checkpoints; pass --resume or use a new run folder"
            )
        self._write_json("run_config.json", {"config": self.config.to_dict(), "run": self.describe()})
        stop = self.total_steps if max_steps is None else min(self.total_steps, self.step + max_steps)
        self.log(f"stage {self.stage_number}: steps {self.step} -> {stop} of {self.total_steps}; {self.device.describe()}")

        previous = {s: signal.signal(s, self._on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        feeder = StepFeeder(self.prepare_step, self.step, stop)
        started = time.perf_counter()
        window_tokens, window_time, window_loss = 0.0, time.perf_counter(), []
        try:
            if evaluate and self.step == 0 and self.stage.eval_every > 0:
                self._evaluate()
            while self.step < stop:
                step_started = time.perf_counter()
                prepared = feeder.next()
                assert prepared.index == self.step
                stats = self.train_step(prepared)
                self.step += 1
                stats.update(
                    step=self.step,
                    source=prepared.batch.source,
                    seconds=time.perf_counter() - step_started,
                )
                self._append("train_log.jsonl", stats)
                window_tokens += stats["padded_tokens"]
                window_loss.append(stats["loss"])
                if self.step % 20 == 0 or self.step == stop:
                    now = time.perf_counter()
                    self.log(
                        f"step {self.step}/{self.total_steps}  loss {np.mean(window_loss):.4f}  "
                        f"lr {stats['lr']:.2e}  {window_tokens / (now - window_time):,.0f} tok/s  "
                        f"peak {peak_memory_gb(self.device.device):.1f} GB"
                    )
                    window_tokens, window_time, window_loss = 0.0, now, []
                if evaluate and self.stage.eval_every > 0 and self.step % self.stage.eval_every == 0:
                    self._evaluate()
                if self.step % self.stage.save_every == 0 or self.step == stop or self._stop_requested:
                    self.save_checkpoint()
                if self._stop_requested:
                    self.log(f"stopped on request at step {self.step}; resume with --resume")
                    break
            if self.step == self.total_steps:
                if evaluate and self.stage.eval_every > 0 and self.step % self.stage.eval_every != 0:
                    self._evaluate()
                self._save_adapter(self.run_dir / "final")
        finally:
            feeder.close()
            for number, handler in previous.items():
                signal.signal(number, handler)
        return {"step": self.step, "seconds": time.perf_counter() - started, "best_dev": self.best_dev}

    def _on_signal(self, number: int, frame: Any) -> None:
        self._stop_requested = True

    # -- evaluation --------------------------------------------------------

    def _evaluate(self) -> dict[str, float]:
        started = time.perf_counter()
        if self._dev_tasks is None:
            self._dev_tasks = build_dev_tasks(self.cache, self.config)
        result = evaluate_dev(self.base.encoder, self._dev_tasks, self.device)
        every = self.config.dev.benchmark_every
        if every > 0 and self.config.dev.benchmark_tasks and self.step % every == 0:
            if self._benchmark is None:
                self._benchmark = Benchmark(self.config, self.tokenizer)
            result.update(
                self._benchmark.evaluate(self.base.encoder, self.config.dev.benchmark_tasks, self.device)
            )
        result.update(step=self.step, seconds=time.perf_counter() - started)
        self._append("eval_log.jsonl", result)
        dev = result["dev_ndcg10"]
        self.log(f"eval at step {self.step}: dev nDCG@10 {dev:.4f} ({result['seconds']:.0f} s)")
        if self.best_dev is None or dev > self.best_dev:
            self.best_dev = dev
            self._save_adapter(self.run_dir / "best", {"dev_ndcg10": dev})
        return result

    # -- checkpoints -------------------------------------------------------

    def _save_adapter(self, directory: Path, extra: dict[str, Any] | None = None) -> None:
        temporary = directory.with_name(directory.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        details = {"stage": self.stage_number, "step": self.step, "plan_digest": self.plan.digest}
        details.update(extra or {})
        save_adapter(temporary, self.adapter, self.base, embedding_contract(self.config), details)
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(temporary, directory)

    def save_checkpoint(self) -> Path:
        directory = self.run_dir / f"{CHECKPOINT_PREFIX}{self.step:07d}"
        self._save_adapter(directory)
        state = {
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
            "best_dev": self.best_dev,
            "plan_digest": self.plan.digest,
            "config_digest": self._config_digest(),
            "cache_digest": self.cache.digest,
        }
        temporary = directory / (STATE_FILE + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, directory / STATE_FILE)
        for old in self._checkpoints()[: -self.stage.keep_last]:
            shutil.rmtree(old)
        return directory

    def _checkpoints(self) -> list[Path]:
        """Complete checkpoints, oldest first."""
        found = [p for p in self.run_dir.glob(f"{CHECKPOINT_PREFIX}*") if (p / STATE_FILE).is_file()]
        return sorted(found, key=lambda path: int(path.name[len(CHECKPOINT_PREFIX) :]))

    def _resume(self) -> None:
        checkpoints = self._checkpoints()
        if not checkpoints:
            raise EmbeddingAdapterError(f"nothing to resume in {self.run_dir}")
        directory = checkpoints[-1]
        state = torch.load(directory / STATE_FILE, map_location="cpu", weights_only=True)
        for key, current in (
            ("plan_digest", self.plan.digest),
            ("config_digest", self._config_digest()),
            ("cache_digest", self.cache.digest),
        ):
            if state[key] != current:
                raise EmbeddingAdapterError(
                    f"cannot resume {directory}: {key} was {state[key]}, now {current}. "
                    f"The data or the settings changed since the run started."
                )
        self.adapter.load_state_dict(load_file(str(directory / ADAPTER_WEIGHTS)))
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["step"])
        self.best_dev = state["best_dev"]
        self.log(f"resumed from {directory.name}")

    # -- files -------------------------------------------------------------

    def _append(self, name: str, record: dict[str, Any]) -> None:
        with (self.run_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        with (self.run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
