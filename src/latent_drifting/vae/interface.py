"""VAE protocol and tensor-shape helpers."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class LatentAutoencoder(Protocol):
    scale_factor: float

    def encode(self, images: torch.Tensor, sample_posterior: bool = True) -> torch.Tensor: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


class FrozenAutoencoder(nn.Module):
    """Base class ensuring tokenizer weights stay frozen while decoder input gradients flow."""

    scale_factor: float

    def freeze(self) -> None:
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenAutoencoder:
        super().train(False)
        return self
