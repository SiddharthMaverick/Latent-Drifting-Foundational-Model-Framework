"""Deterministic 8x8 RGB patchification used by the paper's pixel-MAE encoder."""

from __future__ import annotations

import torch
from torch import nn


class PixelPatchifier(nn.Module):
    """Rearrange RGB 256x256 images into 32x32x192 feature grids."""

    def __init__(self, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Pixel patchifier expects RGB NCHW images")
        batch, channels, height, width = images.shape
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError("Image dimensions must be divisible by pixel-MAE patch size")
        grid_h, grid_w = height // patch, width // patch
        return (
            images.reshape(batch, channels, grid_h, patch, grid_w, patch)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(batch, channels * patch * patch, grid_h, grid_w)
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4 or patches.shape[1] != 3 * self.patch_size**2:
            raise ValueError("Patch tensor must have RGB patch channels")
        batch, _, grid_h, grid_w = patches.shape
        patch = self.patch_size
        return (
            patches.reshape(batch, 3, patch, patch, grid_h, grid_w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(batch, 3, grid_h * patch, grid_w * patch)
        )
