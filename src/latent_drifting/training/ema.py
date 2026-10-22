"""Multi-decay exponential moving averages for generator weights."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class MultiEMA:
    def __init__(self, model: nn.Module, decays: Iterable[float]) -> None:
        self.decays = tuple(float(value) for value in decays)
        if not self.decays or any(not 0 < value < 1 for value in self.decays):
            raise ValueError("EMA decays must be in (0, 1)")
        self.shadow: dict[float, dict[str, torch.Tensor]] = {
            decay: {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
                if torch.is_floating_point(value)
            }
            for decay in self.decays
        }
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for decay, values in self.shadow.items():
            for name, average in values.items():
                current = state[name].detach().to(device=average.device, dtype=average.dtype)
                average.lerp_(current, 1.0 - decay)
        self.updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module, decay: float) -> dict[str, torch.Tensor]:
        if decay not in self.shadow:
            raise KeyError(f"No EMA state for decay {decay}")
        state = model.state_dict()
        backup = {name: state[name].detach().clone() for name in self.shadow[decay]}
        for name, value in self.shadow[decay].items():
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
        model.load_state_dict(state)
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        for name, value in backup.items():
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
        model.load_state_dict(state)

    def state_dict(self) -> dict[str, Any]:
        return {"decays": self.decays, "shadow": self.shadow, "updates": self.updates}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if tuple(state["decays"]) != self.decays:
            raise ValueError("EMA decay set differs from checkpoint")
        self.shadow = {
            float(decay): {name: value.detach().clone() for name, value in values.items()}
            for decay, values in state["shadow"].items()
        }
        self.updates = int(state["updates"])
