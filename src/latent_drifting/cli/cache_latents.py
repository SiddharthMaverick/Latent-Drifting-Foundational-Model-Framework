"""Encode an image folder into validated sharded SD-VAE latents."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_drifting.config.loaders import load_config
from latent_drifting.data.image_dataset import ImageFolderDataset
from latent_drifting.data.transforms import ImageTransform
from latent_drifting.data.vae_cache import cache_latents
from latent_drifting.cli.common import resolve_device, torch_dtype
from latent_drifting.utils.reproducibility import seed_everything
from latent_drifting.vae.diffusers_backend import DiffusersAutoencoder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("sample", "mode"), default="sample")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)
    seed_everything(config.experiment.seed)
    dataset = ImageFolderDataset(
        config.data.root,
        ImageTransform(config.data.image_size, config.data.random_resized_crop_scale),
    )
    if len(dataset.class_to_idx) != config.data.num_classes:
        raise ValueError(
            f"Dataset has {len(dataset.class_to_idx)} classes; config says {config.data.num_classes}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.workers > 0,
    )
    vae = DiffusersAutoencoder.from_pretrained(
        config.vae.model_id,
        revision=config.vae.revision,
        dtype=torch_dtype(config.vae.dtype),
        scale_factor=config.vae.scale_factor,
    ).to(device)
    manifest = cache_latents(
        loader,
        vae,
        args.output or config.data.latent_cache,
        device,
        shard_size=args.shard_size,
        sample_posterior=args.mode == "sample",
        metadata={"dataset": dataset.manifest(), "vae_model_id": config.vae.model_id},
    )
    print(f"Cached {manifest['sample_count']} latents across {len(manifest['shards'])} shards")


if __name__ == "__main__":
    main()
