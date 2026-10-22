"""Pixel-space MAE adapter around the shared 32x32 convolutional encoder-decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from latent_drifting.features.pixel import PixelPatchifier
from latent_drifting.mae.model import LatentMAE, MAEOutput


class PixelMAE(nn.Module):
    """Apply 8x8 pixel patchification, MAE reconstruction, and RGB unpatchification."""

    def __init__(
        self,
        base_width: int = 256,
        blocks_per_stage: tuple[int, int, int, int] = (3, 4, 6, 3),
        num_classes: int | None = None,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.patchifier = PixelPatchifier(patch_size)
        self.model = LatentMAE(3 * patch_size**2, base_width, blocks_per_stage, num_classes)
        self.patch_size = patch_size

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def classifier(self):
        return self.model.classifier

    def forward(
        self,
        inputs: torch.Tensor,
        mask_ratio: float = 0.5,
        mask_patch_size: int = 2,
        apply_random_mask: bool = True,
        generator: torch.Generator | None = None,
    ) -> MAEOutput:
        patch_input = self.patchifier(inputs)
        output = self.model(
            patch_input,
            mask_ratio=mask_ratio,
            mask_patch_size=mask_patch_size,
            apply_random_mask=apply_random_mask,
            generator=generator,
        )
        reconstruction = self.patchifier.unpatchify(output.reconstruction)
        mask = F.interpolate(output.mask, size=inputs.shape[-2:], mode="nearest")
        return MAEOutput(reconstruction, output.features, mask)

    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.classify(self.patchifier(inputs))

    def encode_pyramid(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder.forward_pyramid(self.patchifier(inputs))
