"""RMS normalization used by the drifting DiT backbone."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension)) if affine else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.float() * torch.rsqrt(inputs.float().square().mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            values = values * self.weight.float()
        return values.to(dtype=inputs.dtype)
