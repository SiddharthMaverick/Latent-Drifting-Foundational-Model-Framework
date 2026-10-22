"""Deterministic one-forward sampling and image serialization."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from latent_drifting.vae.interface import LatentAutoencoder


class Sampler:
    def __init__(
        self,
        generator: nn.Module,
        device: torch.device | str,
        vae: LatentAutoencoder | None = None,
    ) -> None:
        self.generator = generator.to(device).eval()
        self.device = torch.device(device)
        self.vae = vae
        if vae is not None and isinstance(vae, nn.Module):
            vae.to(device).eval()

    @torch.inference_mode()
    def generate(
        self,
        labels: torch.Tensor,
        alpha: float | torch.Tensor = 1.0,
        seed: int = 0,
        decode: bool = True,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        labels = labels.to(device=self.device, dtype=torch.long)
        config = getattr(self.generator, "config")
        noise_generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            (labels.numel(), config.input_channels, config.image_size, config.image_size),
            device=self.device,
            dtype=output_dtype,
            generator=noise_generator,
        )
        style_indices = None
        if getattr(config, "use_style_tokens", False):
            style_indices = torch.randint(
                config.style_codebook_size,
                (labels.numel(), config.style_tokens),
                device=self.device,
                generator=noise_generator,
            )
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device=self.device, dtype=torch.float32)
        generated = self.generator(noise, labels, alpha, style_indices)
        if config.input_channels == 4 and decode:
            if self.vae is None:
                raise RuntimeError("Latent generation requested pixel decoding but no VAE was supplied")
            generated = self.vae.decode(generated)
        return generated

    @torch.inference_mode()
    def batches(
        self,
        labels: torch.Tensor,
        batch_size: int,
        alpha: float = 1.0,
        seed: int = 0,
        decode: bool = True,
    ) -> Iterator[torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for batch_index, label_batch in enumerate(labels.split(batch_size)):
            yield self.generate(label_batch, alpha, seed + batch_index, decode)


def save_image_tensor(image: torch.Tensor, path: str | Path) -> None:
    """Save one ``[-1,1]`` CHW/RGB sample as an 8-bit PNG."""
    if image.ndim != 3 or image.shape[0] not in {1, 3}:
        raise ValueError("Expected one CHW image with one or three channels")
    array = image.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).round().byte()
    array = array.permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array)).save(target)
