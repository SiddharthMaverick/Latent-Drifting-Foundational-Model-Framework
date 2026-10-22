"""U-Net-like reconstruction head for the convolutional masked autoencoder."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, channels), channels)


class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        values = F.interpolate(values, size=size, mode="bilinear", align_corners=False)
        return self.layers(values)


class MAEDecoder(nn.Module):
    def __init__(self, encoder_channels: tuple[int, int, int, int], output_channels: int) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.initial = ReconstructionBlock(c4, c4)
        self.up3 = ReconstructionBlock(c4 + c3, c3)
        self.up2 = ReconstructionBlock(c3 + c2, c2)
        self.up1 = ReconstructionBlock(c2 + c1, c1)
        self.refine = ReconstructionBlock(c1, c1)
        self.output = nn.Conv2d(c1, output_channels, kernel_size=1)

    def forward(self, features: tuple[torch.Tensor, ...], output_size: tuple[int, int]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("MAE decoder expects four ResNet stage outputs")
        f1, f2, f3, f4 = features
        values = self.initial(f4, f4.shape[-2:])
        values = self.up3(torch.cat((F.interpolate(values, size=f3.shape[-2:], mode="bilinear", align_corners=False), f3), 1), f3.shape[-2:])
        values = self.up2(torch.cat((F.interpolate(values, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2), 1), f2.shape[-2:])
        values = self.up1(torch.cat((F.interpolate(values, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1), 1), f1.shape[-2:])
        values = self.refine(values, output_size)
        return self.output(values)
