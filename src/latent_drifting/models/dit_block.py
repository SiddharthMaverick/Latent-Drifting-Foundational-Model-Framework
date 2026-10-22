"""SwiGLU transformer block with QK-Norm, 2D RoPE, and adaLN-Zero."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from latent_drifting.models.norm import RMSNorm
from latent_drifting.models.rope import apply_rope


def _modulate(values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return values * (1 + scale[:, None, :]) + shift[:, None, :]


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, multiplier: float = 8 / 3) -> None:
        super().__init__()
        inner = int(math.ceil(hidden_dim * multiplier / 256) * 256)
        self.gate = nn.Linear(hidden_dim, inner, bias=False)
        self.value = nn.Linear(hidden_dim, inner, bias=False)
        self.output = nn.Linear(inner, hidden_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.gate(inputs)) * self.value(inputs))


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 4:
            raise ValueError("head dimension must be divisible by four for 2D RoPE")
        self.norm_attention = RMSNorm(hidden_dim, affine=False)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attention_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_mlp = RMSNorm(hidden_dim, affine=False)
        self.mlp = SwiGLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.modulation(condition).chunk(6, dim=-1)
        batch, token_count, _ = tokens.shape
        qkv = self.qkv(_modulate(self.norm_attention(tokens), shift_attn, scale_attn))
        qkv = qkv.reshape(batch, token_count, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        cosine, sine = rope
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout.p if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, token_count, self.hidden_dim)
        tokens = tokens + gate_attn[:, None, :] * self.dropout(self.attention_output(attended))
        mlp_input = _modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp)
        tokens = tokens + gate_mlp[:, None, :] * self.dropout(self.mlp(mlp_input))
        return tokens
