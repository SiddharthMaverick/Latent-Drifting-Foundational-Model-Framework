"""Masked autoencoder with a ResNet feature encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from latent_drifting.mae.decoder import MAEDecoder
from latent_drifting.mae.masking import apply_mask, random_patch_mask
from latent_drifting.mae.resnet_encoder import ResNetMAEEncoder


@dataclass(slots=True)
class MAEOutput:
    reconstruction: torch.Tensor
    features: tuple[torch.Tensor, ...]
    mask: torch.Tensor


class LatentMAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        base_width: int = 256,
        blocks_per_stage: tuple[int, int, int, int] = (3, 4, 6, 3),
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ResNetMAEEncoder(input_channels, base_width, blocks_per_stage)
        self.decoder = MAEDecoder(self.encoder.output_channels, input_channels)
        self.classifier = (
            nn.Linear(self.encoder.output_channels[-1], num_classes) if num_classes is not None else None
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask_ratio: float = 0.5,
        mask_patch_size: int = 2,
        apply_random_mask: bool = True,
        generator: torch.Generator | None = None,
    ) -> MAEOutput:
        mask = (
            random_patch_mask(inputs, mask_ratio, mask_patch_size, generator)
            if apply_random_mask
            else torch.zeros_like(inputs[:, :1])
        )
        masked = apply_mask(inputs, mask)
        pyramid = self.encoder.forward_pyramid(masked)
        features = tuple(pyramid[f"stage{index}_out"] for index in range(1, 5))
        reconstruction = self.decoder(features, inputs.shape[-2:])
        return MAEOutput(reconstruction, features, mask)

    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("This LatentMAE was constructed without a classifier head")
        features = self.encoder(inputs)[-1]
        return self.classifier(features.mean(dim=(2, 3)))

    def encode_pyramid(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder.forward_pyramid(inputs)


def masked_reconstruction_loss(output: MAEOutput, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    squared = (output.reconstruction - inputs).square() * output.mask
    return squared.sum() / output.mask.sum().clamp_min(eps) / inputs.shape[1]
