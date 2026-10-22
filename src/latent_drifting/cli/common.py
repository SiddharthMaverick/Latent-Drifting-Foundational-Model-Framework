"""Shared CLI setup helpers."""

from __future__ import annotations

import torch

from latent_drifting.config.schema import AppConfig


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the selected PyTorch runtime")
    return device


def torch_dtype(name: str) -> torch.dtype:
    options = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
               "float32": torch.float32, "fp32": torch.float32}
    try:
        return options[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported floating-point dtype: {name}") from error


def app_generator_kwargs(config: AppConfig) -> dict[str, object]:
    section = config.generator
    return {
        "variant": section.variant,
        "input_channels": section.input_channels,
        "image_size": section.image_size,
        "patch_size": section.patch_size,
        "hidden_dim": section.hidden_dim,
        "depth": section.depth,
        "num_heads": section.num_heads,
        "num_classes": section.num_classes,
        "register_tokens": section.register_tokens,
        "style_tokens": section.style_tokens,
        "style_codebook_size": section.style_codebook_size,
        "use_style_tokens": section.use_style_tokens,
        "dropout": section.dropout,
    }
