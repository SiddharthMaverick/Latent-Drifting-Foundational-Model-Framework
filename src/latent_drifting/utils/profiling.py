"""Low-overhead phase timers and CUDA peak-memory reporting."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator

import torch


@contextlib.contextmanager
def phase_timer(name: str, synchronize_cuda: bool = False) -> Iterator[dict[str, float]]:
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    result = {"name": name}
    started = time.perf_counter()
    try:
        yield result
    finally:
        if synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - started


def peak_cuda_memory_mb(device: int | torch.device | None = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)
