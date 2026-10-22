"""Frozen multi-scale feature interface over the latent ResNet encoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from latent_drifting.features.descriptors import extract_descriptors
from latent_drifting.features.pixel import PixelPatchifier
from latent_drifting.mae.resnet_encoder import ResNetMAEEncoder


class FeatureEncoder(nn.Module):
    """Feature adapter that freezes encoder parameters but preserves input gradients."""

    def __init__(
        self,
        encoder: ResNetMAEEncoder,
        include_input_statistics: bool = True,
        input_projection: PixelPatchifier | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.include_input_statistics = include_input_statistics
        self.input_projection = input_projection
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> FeatureEncoder:
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_input = self.input_projection(inputs) if self.input_projection is not None else inputs
        maps = self.encoder.forward_pyramid(feature_input)
        return extract_descriptors(
            maps,
            feature_input if self.include_input_statistics else None,
        )

    def load_encoder_state(self, state: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        self.encoder.load_state_dict(state, strict=strict)

