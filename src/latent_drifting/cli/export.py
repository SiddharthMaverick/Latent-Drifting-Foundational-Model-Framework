"""Package EMA or raw generator weights and optionally export a runtime format."""

from __future__ import annotations

import argparse

import torch

from latent_drifting.cli.common import resolve_device
from latent_drifting.config.loaders import load_config
from latent_drifting.deployment.artifact import DeploymentArtifact
from latent_drifting.deployment.export_onnx import export_onnx
from latent_drifting.deployment.export_torchscript import export_torchscript
from latent_drifting.models.generator import DriftingGenerator
from latent_drifting.training.ema import MultiEMA


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--format", choices=("artifact", "torchscript", "onnx"), default="artifact")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)
    generator = DriftingGenerator(config.generator).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["model"])
    if args.ema_decay is not None:
        ema = MultiEMA(generator, config.optimization.ema_decays)
        if checkpoint.get("ema") is None:
            raise ValueError("Checkpoint does not contain EMA weights")
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(generator, args.ema_decay)
    if args.format == "artifact":
        artifact = DeploymentArtifact.from_generator(
            generator,
            vae_reference={"model_id": config.vae.model_id, "scale_factor": config.vae.scale_factor}
            if config.generator.input_channels == 4 else None,
            metadata={"training_step": checkpoint.get("step"), "ema_decay": args.ema_decay},
        )
        artifact.save(args.output)
    elif args.format == "torchscript":
        export_torchscript(generator, args.output, device=device)
    else:
        export_onnx(generator, args.output, opset=config.deployment.opset, device=device)
    print(f"Exported {args.format} artifact to {args.output}")


if __name__ == "__main__":
    main()
