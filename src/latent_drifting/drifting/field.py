"""Public drifting-field wrapper."""

from __future__ import annotations

import torch

from latent_drifting.drifting.bilateral_kernel import bilateral_drift


class DriftingField:
    def __init__(self, chunk_size: int = 1024) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def __call__(
        self,
        x: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float,
        negative_weights: torch.Tensor | None = None,
        exclude_aligned_self: bool = False,
        self_exclusion_count: int | None = None,
    ) -> torch.Tensor:
        return bilateral_drift(
            x,
            positives,
            negatives,
            temperature,
            negative_weights=negative_weights,
            exclude_aligned_self=exclude_aligned_self,
            self_exclusion_count=self_exclusion_count,
            chunk_size=self.chunk_size,
        )
