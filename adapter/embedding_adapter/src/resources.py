"""
Choosing and preparing the device.

GPU: one process owns the card. Starting next to another GPU job is refused
unless the caller says the other job is paused (`allow_busy_gpu`), because on
this vGPU two active jobs slow each other by an order of magnitude. A memory
cap keeps this process inside its share when the card is shared with a paused
job. bf16 autocast, TF32 matmuls.

CPU: float32 (the CPUs here have no bf16 instructions, so autocast would only
be slower) and at most `CPU_THREADS` threads. Measured on the 42-core host with
the released base: 12 threads 211 padded tok/s, 24 threads 301, 32 threads 302,
38 threads 254, and 42 threads next to another job 26 to 95. More threads than
that only wait on each other.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import torch

from .config import EmbeddingConfig
from .errors import ResourceError


@dataclass(frozen=True)
class DevicePlan:
    device: torch.device
    bf16: bool
    threads: int
    memory_gb: float
    chunk_tokens: int  # padded tokens per chunk with gradients
    encode_tokens: int  # padded tokens per batch without gradients

    @property
    def is_gpu(self) -> bool:
        return self.device.type == "cuda"

    def describe(self) -> str:
        precision = "bf16" if self.bf16 else "fp32"
        if self.is_gpu:
            cap = f"{self.memory_gb:g} GB cap" if self.memory_gb else "no memory cap"
            return f"GPU ({torch.cuda.get_device_name(0)}), {precision}, {cap}"
        return f"CPU, {self.threads} threads, {precision}"


CPU_THREADS = 24


def available_cores() -> int:
    return len(os.sched_getaffinity(0))


def plan_device(
    device: str,
    config: EmbeddingConfig,
    *,
    gpu_memory_gb: float | None = None,
    allow_busy_gpu: bool = False,
) -> DevicePlan:
    """Decide how to run on `device` ("gpu" or "cpu") and apply it to this process."""
    resources = config.resources
    precision = config.optim.precision
    if device == "cpu":
        if precision == "bf16":
            raise ResourceError("optim.precision is bf16, which the CPU path does not support; use auto")
        threads = resources.cpu_threads or min(CPU_THREADS, available_cores())
        torch.set_num_threads(threads)
        return DevicePlan(
            device=torch.device("cpu"),
            bf16=False,
            threads=threads,
            memory_gb=0.0,
            chunk_tokens=resources.cpu_chunk_tokens,
            encode_tokens=resources.cpu_chunk_tokens,
        )
    if device != "gpu":
        raise ResourceError(f"device must be 'gpu' or 'cpu', got {device!r}")

    if not torch.cuda.is_available():
        raise ResourceError("no CUDA device is visible; use --device cpu")
    others = other_gpu_processes()
    if others and not allow_busy_gpu:
        listing = ", ".join(f"pid {pid} ({memory} MiB)" for pid, memory in others)
        raise ResourceError(
            f"the GPU is in use by {listing}. Two active jobs slow each other badly on this "
            f"card. Wait for it, or pause it and pass --allow-busy-gpu."
        )
    memory_gb = resources.gpu_memory_gb if gpu_memory_gb is None else gpu_memory_gb
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if memory_gb:
        torch.cuda.set_per_process_memory_fraction(min(1.0, memory_gb / total_gb), 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # The GPU process needs few CPU threads itself; batches are built by a helper thread.
    torch.set_num_threads(min(8, available_cores()))
    return DevicePlan(
        device=torch.device("cuda", 0),
        bf16=precision in ("auto", "bf16"),
        threads=torch.get_num_threads(),
        memory_gb=float(memory_gb),
        chunk_tokens=resources.chunk_tokens,
        encode_tokens=resources.encode_tokens,
    )


def other_gpu_processes() -> list[tuple[int, int]]:
    """(pid, MiB) of every other process with a GPU context, via nvidia-smi (no CUDA init)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceError(f"cannot query GPU processes with nvidia-smi: {exc}") from exc
    return parse_gpu_processes(result.stdout, own_pid=os.getpid())


def parse_gpu_processes(output: str, own_pid: int) -> list[tuple[int, int]]:
    processes = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        memory = int(parts[1]) if parts[1].isdigit() else 0
        if pid != own_pid:
            processes.append((pid, memory))
    return processes


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 2**30


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
