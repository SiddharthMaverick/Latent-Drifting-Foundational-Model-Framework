"""Export a generator with explicit noise/label/alpha/style inputs to ONNX."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from latent_drifting.deployment.export_torchscript import _TraceableGenerator
from latent_drifting.models.generator import DriftingGenerator


def export_onnx(
    generator: DriftingGenerator,
    path: str | Path,
    batch_size: int = 1,
    opset: int = 18,
    device: torch.device | str = "cpu",
) -> Path:
    generator = generator.to(device).eval()
    config = generator.config
    inputs = (
        torch.randn(batch_size, config.input_channels, config.image_size, config.image_size, device=device),
        torch.zeros(batch_size, dtype=torch.long, device=device),
        torch.ones(batch_size, device=device),
        torch.zeros(batch_size, config.style_tokens, dtype=torch.long, device=device),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _TraceableGenerator(generator),
        inputs,
        target,
        input_names=["noise", "labels", "alpha", "style_indices"],
        output_names=["samples"],
        dynamic_axes={
            "noise": {0: "batch"},
            "labels": {0: "batch"},
            "alpha": {0: "batch"},
            "style_indices": {0: "batch"},
            "samples": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return target
