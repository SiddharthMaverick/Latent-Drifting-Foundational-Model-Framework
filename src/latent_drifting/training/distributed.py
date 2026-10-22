"""Distributed initialization, class ownership, and DDP/FSDP wrapping."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


def initialize_distributed(backend: str | None = None) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        chosen_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=chosen_backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def class_indices_for_rank(num_classes: int, rank: int, world_size: int) -> list[int]:
    if num_classes < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid class partition dimensions")
    return [index for index in range(num_classes) if index % world_size == rank]


def wrap_distributed(
    model: nn.Module,
    strategy: str = "ddp",
    device_id: int | None = None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    if not dist.is_available() or not dist.is_initialized():
        return model
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model,
            device_ids=[device_id] if device_id is not None and torch.cuda.is_available() else None,
            find_unused_parameters=find_unused_parameters,
        )
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        return FullyShardedDataParallel(
            model,
            device_id=torch.device("cuda", device_id) if device_id is not None else None,
            use_orig_params=True,
        )
    raise ValueError("strategy must be 'ddp' or 'fsdp'")


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result


def all_reduce_metrics(metrics: dict[str, float]) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    names = sorted(metrics)
    values = torch.tensor([metrics[name] for name in names], dtype=torch.float64)
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        values = values.cuda()
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {name: float(value) for name, value in zip(names, values.cpu().tolist(), strict=True)}


def rank_zero_only(function: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not dist.is_initialized() or dist.get_rank() == 0:
            return function(*args, **kwargs)
        return None

    return wrapped
