"""ResNet-34-style GroupNorm encoder used by latent-MAE."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, channels), channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)
        self.activation = nn.ReLU(inplace=True)
        self.projection: nn.Module
        if stride != 1 or in_channels != channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                _group_norm(channels),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        values = self.activation(self.norm1(self.conv1(inputs)))
        values = self.norm2(self.conv2(values))
        return self.activation(values + residual)


class ResNetMAEEncoder(nn.Module):
    """Convolutional encoder emitting four scales and intermediate block taps."""

    def __init__(
        self,
        input_channels: int = 4,
        base_width: int = 256,
        blocks_per_stage: tuple[int, int, int, int] = (3, 4, 6, 3),
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != 4 or any(count < 1 for count in blocks_per_stage):
            raise ValueError("blocks_per_stage must contain four positive integers")
        self.input_channels = input_channels
        self.base_width = base_width
        self.blocks_per_stage = blocks_per_stage
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_width, 3, stride=1, padding=1, bias=False),
            _group_norm(base_width),
            nn.ReLU(inplace=True),
        )
        channels = [base_width, base_width * 2, base_width * 4, base_width * 8]
        current = base_width
        stages: list[nn.Sequential] = []
        for stage_index, (width, count) in enumerate(zip(channels, blocks_per_stage, strict=True)):
            blocks: list[nn.Module] = []
            for block_index in range(count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(BasicBlock(current, width, stride))
                current = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.output_channels = tuple(channels)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward_pyramid(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected NCHW with {self.input_channels} channels, got {tuple(inputs.shape)}"
            )
        values = self.stem(inputs)
        result: dict[str, torch.Tensor] = {}
        for stage_index, stage in enumerate(self.stages, start=1):
            for block_index, block in enumerate(stage, start=1):
                values = block(values)
                if block_index == len(stage):
                    result[f"stage{stage_index}_out"] = values
                elif block_index % 2 == 0:
                    result[f"stage{stage_index}_block{block_index}"] = values
        return result

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        pyramid = self.forward_pyramid(inputs)
        return tuple(pyramid[f"stage{i}_out"] for i in range(1, 5))

