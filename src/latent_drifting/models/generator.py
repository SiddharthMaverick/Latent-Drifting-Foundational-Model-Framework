"""Single-pass DiT generator for latent or pixel-space Drifting Models."""

from __future__ import annotations

import torch
from torch import nn

from latent_drifting.config.schema import GeneratorConfig
from latent_drifting.models.conditioning import ConditionEmbedder
from latent_drifting.models.dit_block import DiTBlock
from latent_drifting.models.patchify import FinalLayer, PatchEmbed
from latent_drifting.models.rope import build_2d_rope


class DriftingGenerator(nn.Module):
    """Map Gaussian noise, class labels, and CFG alpha to one generated sample."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        if config.hidden_dim // config.num_heads % 4:
            raise ValueError("Each attention head dimension must be divisible by four")
        self.config = config
        self.patch_embed = PatchEmbed(config.input_channels, config.hidden_dim, config.patch_size)
        self.conditioner = ConditionEmbedder(
            config.hidden_dim,
            config.num_classes,
            config.style_codebook_size,
            config.style_tokens,
            config.use_style_tokens,
        )
        self.register_tokens = nn.Parameter(torch.randn(1, config.register_tokens, config.hidden_dim) * 0.02)
        self.register_positions = nn.Parameter(torch.randn(1, config.register_tokens, config.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [DiTBlock(config.hidden_dim, config.num_heads, config.dropout) for _ in range(config.depth)]
        )
        self.final_layer = FinalLayer(config.hidden_dim, config.input_channels, config.patch_size)
        self._initialize_patch_projection()

    def _initialize_patch_projection(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed.projection.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.projection.bias)

    @classmethod
    def from_variant(cls, variant: str, num_classes: int = 1000, **overrides: object) -> DriftingGenerator:
        variants = {
            "B/2": (768, 12, 12, 2, 4, 32),
            "L/2": (1024, 24, 16, 2, 4, 32),
            "B/16": (768, 12, 12, 16, 3, 256),
            "L/16": (1024, 24, 16, 16, 3, 256),
        }
        if variant not in variants:
            raise ValueError(f"Unsupported generator variant: {variant}")
        hidden, depth, heads, patch, channels, image_size = variants[variant]
        values: dict[str, object] = {
            "variant": variant,
            "input_channels": channels,
            "image_size": image_size,
            "patch_size": patch,
            "hidden_dim": hidden,
            "depth": depth,
            "num_heads": heads,
            "num_classes": num_classes,
        }
        values.update(overrides)
        return cls(GeneratorConfig(**values))

    @property
    def null_label(self) -> int:
        return self.config.num_classes

    def forward(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor | float,
        style_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, got {noise.shape[1]}"
            )
        if labels.shape != (noise.shape[0],):
            raise ValueError("labels must have shape [batch]")
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.full((noise.shape[0],), float(alpha), device=noise.device)
        elif alpha.ndim == 0:
            alpha = alpha.expand(noise.shape[0])
        tokens, grid = self.patch_embed(noise)
        condition, style_indices = self.conditioner(labels, alpha, style_indices)
        registers = self.register_tokens + self.register_positions + condition[:, None, :]
        tokens = torch.cat((registers, tokens), dim=1)
        rope = build_2d_rope(
            grid[0],
            grid[1],
            self.config.hidden_dim // self.config.num_heads,
            register_tokens=self.config.register_tokens,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for block in self.blocks:
            tokens = block(tokens, condition, rope)
        return self.final_layer(
            tokens,
            condition,
            grid,
            self.config.patch_size,
            self.config.input_channels,
            self.config.register_tokens,
        )

    def make_noise(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size,
        )
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
