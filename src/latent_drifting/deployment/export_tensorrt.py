"""Optional TensorRT compilation for explicit-input TorchScript generators."""

from __future__ import annotations

from pathlib import Path

import torch


def compile_tensorrt(
    torchscript_path: str | Path,
    output_path: str | Path,
    input_shapes: dict[str, tuple[int, ...]],
    precision: str = "fp16",
) -> Path:
    try:
        import torch_tensorrt
    except ImportError as error:
        raise RuntimeError("Install torch-tensorrt and a compatible TensorRT runtime to compile") from error
    if precision not in {"fp16", "fp32", "bf16"}:
        raise ValueError("TensorRT precision must be fp16, bf16, or fp32")
    module = torch.jit.load(str(torchscript_path), map_location="cuda").eval()
    trt_inputs = [
        torch_tensorrt.Input(shape=shape, dtype=torch.float32 if name in {"noise", "alpha"} else torch.int64)
        for name, shape in input_shapes.items()
    ]
    precision_type = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    compiled = torch_tensorrt.compile(
        module,
        inputs=trt_inputs,
        enabled_precisions={precision_type},
        ir="torchscript",
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(compiled, str(target))
    return target
