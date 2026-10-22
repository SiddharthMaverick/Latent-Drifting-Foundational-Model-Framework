"""Stable pairwise distances used by the kernel and feature normalizer."""

from __future__ import annotations

import torch


def pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return Euclidean distances for ``[..., N, C]`` and ``[..., M, C]`` tensors."""
    if x.shape[:-2] != y.shape[:-2] or x.shape[-1] != y.shape[-1]:
        raise ValueError(f"Incompatible pairwise operands: {tuple(x.shape)} and {tuple(y.shape)}")
    return torch.cdist(x.float(), y.float(), p=2)


def chunked_pairwise_l2(x: torch.Tensor, y: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return torch.cat(
        [pairwise_l2(chunk, y) for chunk in x.split(chunk_size, dim=-2)], dim=-2
    )
