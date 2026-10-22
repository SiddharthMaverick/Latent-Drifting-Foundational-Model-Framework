"""Lazy adapter for the Stable-Diffusion VAE provided by Diffusers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from latent_drifting.vae.interface import FrozenAutoencoder


class DiffusersAutoencoder(FrozenAutoencoder):
    def __init__(self, model: nn.Module, scale_factor: float = 0.18215) -> None:
        super().__init__()
        self.model = model
        self.scale_factor = float(scale_factor)
        self.freeze()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "stabilityai/sd-vae-ft-mse",
        revision: str | None = None,
        dtype: torch.dtype = torch.float32,
        scale_factor: float | None = None,
    ) -> DiffusersAutoencoder:
        try:
            from diffusers import AutoencoderKL
        except ImportError as error:
            raise RuntimeError("Install latent-drifting[vae] to load a Diffusers VAE") from error
        model = AutoencoderKL.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
        model_scale = getattr(model.config, "scaling_factor", 0.18215)
        return cls(model, scale_factor if scale_factor is not None else model_scale)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode(self, images: torch.Tensor, sample_posterior: bool = True) -> torch.Tensor:
        encoded = self.model.encode(images)
        posterior = encoded.latent_dist
        latents = posterior.sample() if sample_posterior else posterior.mode()
        return latents * self.scale_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        decoded: Any = self.model.decode(latents / self.scale_factor)
        image: torch.Tensor = decoded.sample
        return image

    def save_pretrained_reference(self) -> dict[str, Any]:
        return {
            "backend": "diffusers",
            "model_id": getattr(self.model.config, "_name_or_path", None),
            "scale_factor": self.scale_factor,
        }
