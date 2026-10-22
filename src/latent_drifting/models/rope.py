"""Two-dimensional rotary positional embedding for image-grid tokens."""

from __future__ import annotations

import torch


def _rotate_pairs(values: torch.Tensor) -> torch.Tensor:
    paired = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def build_2d_rope(
    height: int,
    width: int,
    head_dim: int,
    register_tokens: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 4:
        raise ValueError("2D rotary embeddings require head_dim divisible by four")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    axis_dim = head_dim // 2
    inv_frequency = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2, device=device).float() / axis_dim))
    angle_y = y.flatten()[:, None] * inv_frequency[None, :]
    angle_x = x.flatten()[:, None] * inv_frequency[None, :]
    angle = torch.cat((angle_y.repeat_interleave(2, dim=-1), angle_x.repeat_interleave(2, dim=-1)), dim=-1)
    cosine, sine = angle.cos(), angle.sin()
    if register_tokens:
        prefix_cos = torch.ones((register_tokens, head_dim), device=device, dtype=cosine.dtype)
        prefix_sin = torch.zeros_like(prefix_cos)
        cosine = torch.cat((prefix_cos, cosine), dim=0)
        sine = torch.cat((prefix_sin, sine), dim=0)
    return cosine.to(dtype=dtype), sine.to(dtype=dtype)


def apply_rope(values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    if values.shape[-2] != cosine.shape[-2] or values.shape[-1] != cosine.shape[-1]:
        raise ValueError("RoPE sequence or head dimensions do not match the tensor")
    cosine = cosine.to(device=values.device, dtype=values.dtype)[None, None, :, :]
    sine = sine.to(device=values.device, dtype=values.dtype)[None, None, :, :]
    return values * cosine + _rotate_pairs(values) * sine
