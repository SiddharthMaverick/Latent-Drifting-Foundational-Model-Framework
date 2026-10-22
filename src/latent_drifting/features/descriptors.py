"""Convert feature maps into the spatial and statistical vectors used by the paper."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _pooled_vectors(feature: torch.Tensor, patch: int, statistic: str) -> torch.Tensor:
    batch, channels, height, width = feature.shape
    pooled = F.avg_pool2d(feature, kernel_size=patch, stride=patch, ceil_mode=True)
    if statistic == "mean":
        values = pooled
    elif statistic == "std":
        second = F.avg_pool2d(feature.square(), kernel_size=patch, stride=patch, ceil_mode=True)
        values = (second - pooled.square()).clamp_min(1e-6).sqrt()
    else:
        raise ValueError(f"Unsupported pooled statistic: {statistic}")
    return values.flatten(2).transpose(1, 2).reshape(batch, -1, channels)


def extract_descriptors(
    feature_maps: Mapping[str, torch.Tensor],
    input_tensor: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Produce ``[batch, descriptor_count, channels]`` groups for every feature map.

    The location-level descriptors are retained, while global mean/std and 2x2/
    4x4 patch mean/std groups are independent kernel/loss terms. Spatial layout
    and channel count remain explicit in each group's tensor shape.
    """
    descriptors: dict[str, torch.Tensor] = {}
    if not feature_maps:
        raise ValueError("At least one feature map is required")
    batch = next(iter(feature_maps.values())).shape[0]
    for name, feature in feature_maps.items():
        if feature.ndim != 4:
            raise ValueError(f"Feature map {name!r} must have NCHW rank 4, got {tuple(feature.shape)}")
        batch, channels, _, _ = feature.shape
        descriptors[f"{name}/locations"] = feature.flatten(2).transpose(1, 2)
        mean = feature.mean(dim=(2, 3), keepdim=True)
        std = feature.var(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6).sqrt()
        descriptors[f"{name}/global_mean"] = mean.flatten(2).transpose(1, 2)
        descriptors[f"{name}/global_std"] = std.flatten(2).transpose(1, 2)
        for patch in (2, 4):
            descriptors[f"{name}/patch{patch}_mean"] = _pooled_vectors(feature, patch, "mean")
            descriptors[f"{name}/patch{patch}_std"] = _pooled_vectors(feature, patch, "std")
    if input_tensor is not None:
        if input_tensor.ndim != 4:
            raise ValueError("Feature encoder input must be NCHW")
        descriptors["input/squared_channel_mean"] = input_tensor.square().mean(
            dim=(2, 3), keepdim=True
        ).flatten(2).transpose(1, 2)
    for name, value in descriptors.items():
        if value.shape[0] == 0 or value.shape[-1] == 0:
            raise ValueError(f"Descriptor {name!r} has an empty batch or channel axis")
        if value.shape[0] != batch:
            raise RuntimeError(f"Descriptor {name!r} changed the batch dimension")
    return descriptors
