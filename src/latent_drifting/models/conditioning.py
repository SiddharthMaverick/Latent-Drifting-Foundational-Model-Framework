"""Class, guidance-strength, register, and optional random-style conditioning."""

from __future__ import annotations

import math

import torch
from torch import nn


class AlphaEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.projection = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequency = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=alpha.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = alpha.float().reshape(-1, 1) * frequency[None, :]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if embedding.shape[-1] < self.frequency_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.frequency_dim - embedding.shape[-1]))
        return self.projection(embedding)


class ConditionEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        style_codebook_size: int = 64,
        style_tokens: int = 32,
        use_style_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.style_tokens = style_tokens
        self.use_style_tokens = use_style_tokens
        self.class_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.alpha_embedding = AlphaEmbedding(hidden_dim)
        self.style_embedding = (
            nn.Embedding(style_codebook_size, hidden_dim) if use_style_tokens else None
        )

    def forward(
        self,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        style_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if labels.ndim != 1:
            raise ValueError("Class labels must have shape [batch]")
        if alpha.ndim == 0:
            alpha = alpha.expand(labels.shape[0])
        if alpha.shape != labels.shape:
            raise ValueError("alpha must be scalar or have one value per class label")
        if torch.any((labels < 0) | (labels > self.num_classes)):
            raise ValueError(f"Labels must be between zero and null label {self.num_classes}")
        condition = self.class_embedding(labels) + self.alpha_embedding(alpha)
        style: torch.Tensor | None = None
        if self.style_embedding is not None:
            if style_indices is None:
                style_indices = torch.randint(
                    self.style_embedding.num_embeddings,
                    (labels.shape[0], self.style_tokens),
                    device=labels.device,
                )
            if style_indices.shape != (labels.shape[0], self.style_tokens):
                raise ValueError("style_indices shape must be [batch, configured_style_tokens]")
            style = self.style_embedding(style_indices).sum(dim=1) / math.sqrt(self.style_tokens)
            condition = condition + style
        return condition, style_indices if self.style_embedding is not None else None
