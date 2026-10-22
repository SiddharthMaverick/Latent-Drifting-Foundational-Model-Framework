"""Feature-normalized, multi-temperature, detached-target drifting objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from latent_drifting.drifting.cfg import cfg_unconditional_weight
from latent_drifting.drifting.field import DriftingField
from latent_drifting.features.normalization import FeatureNormalizer


@dataclass(slots=True)
class DriftingLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor] | None = None


class DriftingLoss:
    """Calculate the paper's per-feature loss for one class-conditioned group."""

    def __init__(
        self,
        temperatures: tuple[float, ...] = (0.02, 0.05, 0.2),
        chunk_size: int = 1024,
        normalize_features: bool = True,
        normalize_drift: bool = True,
        normalization_momentum: float = 0.99,
        eps: float = 1e-6,
        return_targets: bool = False,
    ) -> None:
        if not temperatures or any(value <= 0 for value in temperatures):
            raise ValueError("temperatures must be a non-empty sequence of positive values")
        self.temperatures = temperatures
        self.field = DriftingField(chunk_size)
        self.normalize_features = normalize_features
        self.normalize_drift = normalize_drift
        self.normalizer = FeatureNormalizer(normalization_momentum, eps)
        self.eps = eps
        self.return_targets = return_targets

    def __call__(
        self,
        generated: Mapping[str, torch.Tensor],
        positives: Mapping[str, torch.Tensor],
        unconditional: Mapping[str, torch.Tensor] | None = None,
        alpha: float | torch.Tensor = 1.0,
        negative_is_generated: bool = True,
    ) -> DriftingLossOutput:
        if set(generated) != set(positives):
            raise ValueError("Generated and positive feature groups must have identical names")
        if unconditional is not None and set(generated) != set(unconditional):
            raise ValueError("Generated and unconditional feature groups must have identical names")
        alpha_value = float(alpha.detach().float().mean()) if isinstance(alpha, torch.Tensor) else float(alpha)
        first_value = next(iter(generated.values()))
        num_generated = first_value.shape[0]
        num_unconditional = 0 if unconditional is None else next(iter(unconditional.values())).shape[0]
        uncond_weight = cfg_unconditional_weight(alpha_value, num_generated, num_unconditional)
        total_loss: torch.Tensor | None = None
        metrics: dict[str, torch.Tensor] = {}
        targets: dict[str, torch.Tensor] = {}

        for name, prediction in generated.items():
            positive = positives[name]
            uncond = None if unconditional is None else unconditional[name]
            if prediction.ndim == 2:
                prediction = prediction[:, None, :]
                positive = positive[:, None, :]
                if uncond is not None:
                    uncond = uncond[:, None, :]
            if prediction.ndim != 3 or positive.ndim != 3:
                raise ValueError(f"Features in group {name!r} must have shape [batch, descriptors, channels]")
            if prediction.shape[1:] != positive.shape[1:]:
                raise ValueError(f"Generated and positive shapes differ for feature {name!r}")
            if uncond is not None and prediction.shape[1:] != uncond.shape[1:]:
                raise ValueError(f"Generated and unconditional shapes differ for feature {name!r}")
            negatives = prediction.detach()
            if self.normalize_features:
                scale = self.normalizer.estimate(
                    name,
                    prediction.detach(),
                    positive,
                    negatives,
                    uncond,
                    uncond_weight,
                )
            else:
                scale = prediction.new_tensor(1.0)
            normalized_prediction = prediction / scale
            normalized_positive = positive.detach() / scale
            normalized_negatives = negatives / scale
            normalized_uncond = None if uncond is None else uncond.detach() / scale
            if normalized_uncond is None:
                kernel_negatives = normalized_negatives
                negative_weights = None
            else:
                kernel_negatives = torch.cat((normalized_negatives, normalized_uncond), dim=0)
                weight_tensor = torch.cat(
                    (
                        torch.ones(num_generated, device=prediction.device),
                        torch.full((num_unconditional,), uncond_weight, device=prediction.device),
                    )
                )
                negative_weights = weight_tensor
            total_drift: torch.Tensor | None = None
            for temperature in self.temperatures:
                drift = self.field(
                    normalized_prediction.detach(),
                    normalized_positive,
                    kernel_negatives,
                    temperature,
                    negative_weights=negative_weights,
                    self_exclusion_count=num_generated if negative_is_generated else 0,
                )
                if self.normalize_drift:
                    drift_scale = drift.float().square().mean().sqrt().clamp_min(self.eps)
                    drift = drift / drift_scale
                    metrics[f"drift_scale/{name}/{temperature:g}"] = drift_scale.detach()
                total_drift = drift if total_drift is None else total_drift + drift
                metrics[f"drift_norm/{name}/{temperature:g}"] = drift.float().square().mean().sqrt().detach()
            assert total_drift is not None
            target = (normalized_prediction + total_drift).detach()
            group_loss = F.mse_loss(normalized_prediction, target)
            total_loss = group_loss if total_loss is None else total_loss + group_loss
            metrics[f"loss/{name}"] = group_loss.detach()
            metrics[f"feature_scale/{name}"] = scale.detach()
            if self.return_targets:
                targets[name] = target

        if total_loss is None:
            raise ValueError("At least one feature group is required")
        metrics["loss"] = total_loss.detach()
        return DriftingLossOutput(total_loss, metrics, targets if self.return_targets else None)

    def state_dict(self) -> dict[str, object]:
        return {"normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.normalizer.load_state_dict(state["normalizer"])  # type: ignore[arg-type]
