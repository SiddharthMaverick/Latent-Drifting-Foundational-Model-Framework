"""Memory-bounded bilateral kernel drift with the paper's two-axis normalization."""

from __future__ import annotations

import math

import torch

from latent_drifting.drifting.distances import pairwise_l2


def _prepare_features(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3:
        raise ValueError(f"Expected [samples, descriptors, channels], got {tuple(value.shape)}")
    return value.detach().float().transpose(0, 1).contiguous()


@torch.no_grad()
def bilateral_drift(
    x: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
    negative_weights: torch.Tensor | None = None,
    exclude_aligned_self: bool = False,
    self_exclusion_count: int | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Estimate the anti-symmetric drift for one descriptor group.

    Inputs use ``[samples, descriptors, channels]``. Spatial descriptors are
    treated independently with one normalization per location and a shared
    temperature. Row and column softmaxes are combined geometrically exactly as
    specified in the paper's Algorithm 2. Query and descriptor chunking bounds
    peak distance-matrix memory without changing either normalization.
    """
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("temperature and chunk_size must be positive")
    x_m = _prepare_features(x)
    pos_m = _prepare_features(positives)
    neg_m = _prepare_features(negatives)
    if x_m.shape[0] != pos_m.shape[0] or x_m.shape[0] != neg_m.shape[0]:
        raise ValueError("All operands must have the same descriptor count")
    if x_m.shape[-1] != pos_m.shape[-1] or x_m.shape[-1] != neg_m.shape[-1]:
        raise ValueError("All operands must have the same channel dimension")
    descriptor_count, query_count, channels = x_m.shape
    pos_count, neg_count = pos_m.shape[1], neg_m.shape[1]
    if pos_count < 1 or neg_count < 1:
        raise ValueError("Positive and negative sets must be non-empty")
    if self_exclusion_count is None:
        self_exclusion_count = neg_count if exclude_aligned_self else 0
    if not 0 <= self_exclusion_count <= min(query_count, neg_count):
        raise ValueError("self_exclusion_count must not exceed query or negative batch size")

    if negative_weights is None:
        negative_weights = torch.ones(neg_count, device=x_m.device, dtype=torch.float32)
    else:
        negative_weights = negative_weights.detach().to(device=x_m.device, dtype=torch.float32).flatten()
        if negative_weights.numel() != neg_count:
            raise ValueError("negative_weights must have one value per negative sample")
        if torch.any(negative_weights < 0):
            raise ValueError("negative_weights cannot be negative")
    pos_log_weights = torch.zeros(pos_count, device=x_m.device, dtype=torch.float32)
    neg_log_weights = torch.where(
        negative_weights > 0,
        negative_weights.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.full_like(negative_weights, -torch.inf),
    )
    candidate_log_weights = torch.cat((pos_log_weights, neg_log_weights))
    candidates = torch.cat((pos_m, neg_m), dim=1)
    temperature_scaled = temperature * channels
    descriptor_chunk = max(1, min(descriptor_count, chunk_size // max(query_count, 1)))
    result = torch.empty_like(x_m)

    for descriptor_start in range(0, descriptor_count, descriptor_chunk):
        descriptor_end = min(descriptor_start + descriptor_chunk, descriptor_count)
        x_part = x_m[descriptor_start:descriptor_end]
        y_part = candidates[descriptor_start:descriptor_end]
        local_m = descriptor_end - descriptor_start
        column_lse = torch.full(
            (local_m, pos_count + neg_count), -torch.inf, device=x_m.device, dtype=torch.float32
        )
        # First pass computes global column log-normalizers over all generated queries.
        for query_start in range(0, query_count, chunk_size):
            query_end = min(query_start + chunk_size, query_count)
            distances = pairwise_l2(x_part[:, query_start:query_end], y_part)
            logits = -distances / temperature_scaled + candidate_log_weights.view(1, 1, -1)
            if self_exclusion_count:
                rows = torch.arange(query_start, query_end, device=x_m.device)
                selected = rows < self_exclusion_count
                logits[:, torch.arange(query_end - query_start, device=x_m.device)[selected], pos_count + rows[selected]] = -torch.inf
            chunk_lse = torch.logsumexp(logits, dim=1)
            column_lse = torch.logaddexp(column_lse, chunk_lse)

        # Second pass forms the geometric row/column-normalized affinity and drift.
        for query_start in range(0, query_count, chunk_size):
            query_end = min(query_start + chunk_size, query_count)
            distances = pairwise_l2(x_part[:, query_start:query_end], y_part)
            logits = -distances / temperature_scaled + candidate_log_weights.view(1, 1, -1)
            if self_exclusion_count:
                rows = torch.arange(query_start, query_end, device=x_m.device)
                selected = rows < self_exclusion_count
                logits[:, torch.arange(query_end - query_start, device=x_m.device)[selected], pos_count + rows[selected]] = -torch.inf
            row_lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            log_affinity = 0.5 * (logits - row_lse - column_lse[:, None, :])
            affinity = torch.where(
                torch.isfinite(candidate_log_weights)[None, None, :],
                torch.exp(log_affinity),
                torch.zeros_like(log_affinity),
            )
            affinity_pos = affinity[..., :pos_count]
            affinity_neg = affinity[..., pos_count:]
            positive_mass = affinity_pos.sum(dim=-1, keepdim=True)
            negative_mass = affinity_neg.sum(dim=-1, keepdim=True)
            attracted = torch.matmul(affinity_pos, y_part[:, :pos_count])
            repelled = torch.matmul(affinity_neg, y_part[:, pos_count:])
            result[descriptor_start:descriptor_end, query_start:query_end] = (
                attracted * negative_mass - repelled * positive_mass
            )

    return result.transpose(0, 1).contiguous().to(dtype=x.dtype)


def bilateral_affinity(
    x: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Dense affinity helper exposed for analysis and diagnostics."""
    x_m = _prepare_features(x)
    y_m = _prepare_features(candidates)
    distances = pairwise_l2(x_m, y_m)
    logits = -distances / (temperature * x.shape[-1])
    log_affinity = 0.5 * (
        torch.log_softmax(logits, dim=-1) + torch.log_softmax(logits, dim=-2)
    )
    return torch.exp(log_affinity).transpose(0, 1)
