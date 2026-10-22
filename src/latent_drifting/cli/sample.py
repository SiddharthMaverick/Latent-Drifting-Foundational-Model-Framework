"""Generate images from a generator-only artifact."""

from __future__ import annotations

import argparse

import torch

from latent_drifting.cli.common import resolve_device, torch_dtype
from latent_drifting.deployment.artifact import DeploymentArtifact
from latent_drifting.evaluation.sampling import Sampler, save_image_tensor
from latent_drifting.vae.diffusers_backend import DiffusersAutoencoder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", default="samples")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vae-model", default=None)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    artifact, generator = DeploymentArtifact.load(args.artifact, device)
    vae = None
    if generator.config.input_channels == 4:
        model_id = args.vae_model or (artifact.vae_reference or {}).get("model_id")
        if model_id is None:
            raise ValueError("Latent artifact has no VAE reference; pass --vae-model")
        vae = DiffusersAutoencoder.from_pretrained(
            model_id,
            dtype=torch_dtype("float32"),
            scale_factor=(artifact.vae_reference or {}).get("scale_factor"),
        ).to(device)
    sampler = Sampler(generator, device, vae)
    labels = torch.full((args.count,), args.class_id, dtype=torch.long)
    images = sampler.generate(labels, args.alpha, args.seed)
    for index, image in enumerate(images):
        save_image_tensor(image, f"{args.output}/sample-{index:05d}.png")
    print(f"Wrote {args.count} images to {args.output}")


if __name__ == "__main__":
    main()
