"""Weighted metric aggregation with optional distributed reduction."""

from __future__ import annotations

from collections import defaultdict

import torch

from latent_drifting.training.distributed import all_reduce_mean


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, values: dict[str, torch.Tensor | float | int], weight: int = 1) -> None:
        for name, value in values.items():
            scalar = float(value.detach().float().mean()) if isinstance(value, torch.Tensor) else float(value)
            self.sums[name] += scalar * weight
            self.counts[name] += weight

    def compute(self, distributed: bool = False, device: torch.device | str = "cpu") -> dict[str, float]:
        result: dict[str, float] = {}
        for name, total in self.sums.items():
            count = self.counts[name]
            pair = torch.tensor([total, count], dtype=torch.float64, device=device)
            if distributed:
                pair = all_reduce_mean(pair) * torch.distributed.get_world_size()
            result[name] = float(pair[0] / pair[1].clamp_min(1))
        return result
