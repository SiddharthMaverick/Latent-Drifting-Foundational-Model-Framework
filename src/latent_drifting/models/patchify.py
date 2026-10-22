"""Image-to-token patch projection and inverse projection."""

from __future__ import annotations

import torch
from torch import nn

from latent_drifting.models.norm import RMSNorm


class PatchEmbed(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            input_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError("Generator input must be NCHW")
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Input dimensions must be divisible by patch size")
        tokens = self.projection(images).flatten(2).transpose(1, 2)
        return tokens, (height // self.patch_size, width // self.patch_size)


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, output_channels: int, patch_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_dim, affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.projection = nn.Linear(hidden_dim, patch_size * patch_size * output_channels)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        grid_size: tuple[int, int],
        patch_size: int,
        output_channels: int,
        register_tokens: int,
    ) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        tokens = self.norm(tokens[:, register_tokens:]) * (1 + scale[:, None, :]) + shift[:, None, :]
        patches = self.projection(tokens)
        batch = patches.shape[0]
        height, width = grid_size
        patches = patches.reshape(batch, height, width, patch_size, patch_size, output_channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, output_channels, height * patch_size, width * patch_size
        )
