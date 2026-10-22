"""Trace a generator with all stochastic inputs explicit."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from latent_drifting.models.generator import DriftingGenerator


class _TraceableGenerator(nn.Module):
    def __init__(self, generator: DriftingGenerator) -> None:
        super().__init__()
        self.generator = generator

    def forward(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        style_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(noise, labels, alpha, style_indices)


def export_torchscript(
    generator: DriftingGenerator,
    path: str | Path,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> Path:
    generator = generator.to(device).eval()
    config = generator.config
    noise = torch.randn(batch_size, config.input_channels, config.image_size, config.image_size, device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    alpha = torch.ones(batch_size, device=device)
    styles = torch.zeros(batch_size, config.style_tokens, dtype=torch.long, device=device)
    traced = torch.jit.trace(_TraceableGenerator(generator), (noise, labels, alpha, styles), strict=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(target))
    return target
