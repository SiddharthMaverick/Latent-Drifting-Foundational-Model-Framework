"""Atomic, restartable full-training checkpoints and generator-only snapshots."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    ema: Any = None,
    scaler: Any = None,
    scheduler: Any = None,
    loss_state: dict[str, Any] | None = None,
    queue_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "step": step,
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "loss": loss_state,
        "queue": queue_state,
        "config": config,
        "rng": capture_rng_state(),
        "extra": extra or {},
    }
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema: Any = None,
    scaler: Any = None,
    scheduler: Any = None,
    loss_object: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if loss_object is not None and payload.get("loss") is not None:
        loss_object.load_state_dict(payload["loss"])
    if restore_rng and payload.get("rng"):
        restore_rng_state(payload["rng"])
    return payload
