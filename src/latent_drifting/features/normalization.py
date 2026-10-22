"""Shared feature-distance scales with checkpointable exponential smoothing."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class FeatureNormalizer:
    def __init__(self, momentum: float = 0.99, eps: float = 1e-6) -> None:
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self.eps = eps
        self.scales: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def estimate(
        self,
        name: str,
        generated: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        unconditional: torch.Tensor | None = None,
        unconditional_weight: float = 0.0,
    ) -> torch.Tensor:
        channels = generated.shape[-1]
        sources = [(positives.detach(), 1.0), (negatives.detach(), 1.0)]
        if unconditional is not None and unconditional_weight > 0:
            sources.append((unconditional.detach(), float(unconditional_weight)))
        # Each descriptor location shares one scale, matching Appendix A.6.
        x = generated.detach().float().permute(1, 0, 2)
        weighted_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        total_mass = 0.0
        for source, weight in sources:
            y = source.float().permute(1, 0, 2)
            weighted_sum = weighted_sum + torch.cdist(x, y).mean() * weight * source.shape[0]
            total_mass += weight * source.shape[0]
        distances = weighted_sum / max(total_mass, 1e-12)
        current = (distances / channels).clamp_min(self.eps).detach()
        previous = self.scales.get(name)
        updated = current if previous is None else self.momentum * previous.to(current) + (1 - self.momentum) * current
        self.scales[name] = updated.detach()
        return updated

    def state_dict(self) -> dict[str, object]:
        return {"momentum": self.momentum, "eps": self.eps, "scales": self.scales}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if float(state["momentum"]) != self.momentum:
            raise ValueError("Normalizer momentum differs from checkpoint")
        self.scales = {key: value.detach().clone() for key, value in state["scales"].items()}  # type: ignore[union-attr]
