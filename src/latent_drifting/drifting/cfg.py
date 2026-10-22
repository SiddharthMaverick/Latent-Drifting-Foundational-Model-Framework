"""Classifier-free guidance sampling and negative-mixture weighting."""

from __future__ import annotations

import math

import torch


def sample_cfg_alpha(
    shape: tuple[int, ...] | int,
    alpha_min: float = 1.0,
    alpha_max: float = 4.0,
    power: float = 3.0,
    sampler: str = "power",
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if alpha_min < 1 or alpha_max < alpha_min:
        raise ValueError("Require 1 <= alpha_min <= alpha_max")
    if isinstance(shape, int):
        shape = (shape,)
    if sampler == "fixed_one":
        return torch.ones(shape, device=device)
    uniform = torch.rand(shape, device=device, generator=generator)
    if sampler == "uniform" or power == 0 or alpha_min == alpha_max:
        return alpha_min + (alpha_max - alpha_min) * uniform
    if sampler != "power":
        raise ValueError("sampler must be power, uniform, or fixed_one")
    if power == 1:
        return alpha_min * torch.exp(uniform * math.log(alpha_max / alpha_min))
    exponent = 1.0 - power
    low = alpha_min**exponent
    high = alpha_max**exponent
    return (low + uniform * (high - low)).pow(1.0 / exponent)


def cfg_unconditional_weight(alpha: float | torch.Tensor, num_generated: int, num_unconditional: int) -> float | torch.Tensor:
    """Map CFG strength to each unconditional negative's weight (paper Appendix A.7)."""
    if num_generated < 2:
        raise ValueError("At least two generated samples are needed for self-excluded negatives")
    if num_unconditional < 0:
        raise ValueError("num_unconditional cannot be negative")
    if num_unconditional == 0:
        return torch.zeros_like(alpha) if isinstance(alpha, torch.Tensor) else 0.0
    return (alpha - 1) * (num_generated - 1) / num_unconditional
