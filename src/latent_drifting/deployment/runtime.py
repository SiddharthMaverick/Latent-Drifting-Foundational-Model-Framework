"""Inference API that performs one generator call per batch."""

from __future__ import annotations

import torch
from torch import nn

from latent_drifting.evaluation.sampling import Sampler
from latent_drifting.vae.interface import LatentAutoencoder


class OneStepRuntime:
    def __init__(
        self,
        generator: nn.Module,
        device: torch.device | str = "cpu",
        vae: LatentAutoencoder | None = None,
    ) -> None:
        self.sampler = Sampler(generator, device, vae)

    def generate(
        self,
        labels: torch.Tensor,
        alpha: float | torch.Tensor = 1.0,
        seed: int = 0,
        decode: bool = True,
    ) -> torch.Tensor:
        return self.sampler.generate(labels, alpha=alpha, seed=seed, decode=decode)
