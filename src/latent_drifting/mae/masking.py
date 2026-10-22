"""Paper-style random 2x2 patch zero masking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_patch_mask(
    inputs: torch.Tensor,
    mask_ratio: float = 0.5,
    patch_size: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if inputs.ndim != 4:
        raise ValueError("MAE input must have NCHW rank 4")
    if not 0 <= mask_ratio <= 1 or patch_size < 1:
        raise ValueError("mask_ratio must be in [0,1] and patch_size positive")
    height, width = inputs.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("Input dimensions must be divisible by the mask patch size")
    grid_h, grid_w = height // patch_size, width // patch_size
    patch_mask = torch.rand(
        (inputs.shape[0], 1, grid_h, grid_w), device=inputs.device, generator=generator
    ) < mask_ratio
    return F.interpolate(patch_mask.float(), scale_factor=patch_size, mode="nearest").to(inputs.dtype)


def apply_mask(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[0] != inputs.shape[0] or mask.shape[-2:] != inputs.shape[-2:]:
        raise ValueError("Mask batch/spatial dimensions must match the input")
    return inputs * (1 - mask)
