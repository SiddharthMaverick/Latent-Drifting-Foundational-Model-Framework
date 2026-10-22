"""Simple nearest-neighbor similarity audit with explicit threshold reporting."""

from __future__ import annotations

import torch

from latent_drifting.evaluation.nearest_neighbors import nearest_neighbors


def audit_memorization(
    generated_features: torch.Tensor,
    training_features: torch.Tensor,
    similarity_threshold: float = 0.98,
    top_k: int = 5,
) -> dict[str, float | int]:
    scores, _ = nearest_neighbors(generated_features, training_features, top_k=top_k)
    best = scores[:, 0]
    return {
        "generated_count": int(best.numel()),
        "nearest_similarity_mean": float(best.mean()),
        "nearest_similarity_p99": float(torch.quantile(best, 0.99)),
        "above_similarity_threshold": int((best >= similarity_threshold).sum()),
        "similarity_threshold": similarity_threshold,
    }
