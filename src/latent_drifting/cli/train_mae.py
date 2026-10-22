"""Pretrain or classifier-finetune the latent-MAE feature encoder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_drifting.cli.common import resolve_device
from latent_drifting.config.loaders import load_config
from latent_drifting.data.image_dataset import ImageFolderDataset
from latent_drifting.data.transforms import ImageTransform
from latent_drifting.data.vae_cache import LatentCacheDataset
from latent_drifting.mae.classifier_finetune import finetune_step
from latent_drifting.mae.model import LatentMAE
from latent_drifting.mae.pixel import PixelMAE
from latent_drifting.mae.pretrain import pretrain_step
from latent_drifting.training.ema import MultiEMA
from latent_drifting.utils.reproducibility import capture_rng_state, restore_rng_state, seed_everything


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--classifier-finetune", action="store_true")
    parser.add_argument("--pixel", action="store_true", help="Train the 8x8-patch pixel-MAE variant")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)
    seed_everything(config.experiment.seed)
    if args.pixel:
        dataset = ImageFolderDataset(
            config.data.root,
            ImageTransform(config.data.image_size, config.data.random_resized_crop_scale),
        )
    else:
        dataset = LatentCacheDataset(args.cache or config.data.latent_cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.data.workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.workers > 0,
    )
    if args.pixel:
        model = PixelMAE(
            config.mae.base_width,
            config.mae.blocks_per_stage,
            config.data.num_classes if args.classifier_finetune else None,
        ).to(device)
    else:
        model = LatentMAE(
            config.mae.input_channels,
            config.mae.base_width,
            config.mae.blocks_per_stage,
            config.data.num_classes if args.classifier_finetune else None,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.mae.learning_rate)
    ema = MultiEMA(model, (config.mae.ema_decay,))
    output = Path(args.output or Path(config.experiment.output_dir) / "latent-mae.pt")
    target_steps = args.steps or (
        config.mae.finetune_steps if args.classifier_finetune else config.mae.pretrain_epochs * len(loader)
    )
    model.train()
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if bool(checkpoint.get("pixel_space", False)) != args.pixel:
            raise ValueError("MAE checkpoint pixel/latent mode does not match --pixel")
        model.load_state_dict(checkpoint.get("raw_model", checkpoint["model"]))
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        if checkpoint.get("rng") is not None:
            restore_rng_state(checkpoint["rng"])
        step = int(checkpoint.get("step", 0))

    def write_checkpoint(current_step: int) -> None:
        raw_state = model.state_dict()
        ema_state = ema.shadow[config.mae.ema_decay]
        averaged_state = {
            name: ema_state[name].detach().cpu().clone()
            if name in ema_state
            else value.detach().cpu().clone()
            for name, value in raw_state.items()
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
        torch.save(
            {
                "format_version": 1,
                "step": current_step,
                "model": averaged_state,
                "raw_model": {name: value.detach().cpu().clone() for name, value in raw_state.items()},
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "config": config.to_dict(),
                "pixel_space": args.pixel,
                "rng": capture_rng_state(),
            },
            temporary,
        )
        temporary.replace(output)

    while step < target_steps:
        for latents, labels in loader:
            latents, labels = latents.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if args.classifier_finetune:
                loss, metrics = finetune_step(
                    model,
                    optimizer,
                    latents,
                    labels,
                    step,
                    config.mae.classifier_warmup_steps,
                    config.mae.classifier_weight,
                    config.mae.mask_ratio,
                    config.mae.mask_patch_size,
                )
            else:
                loss = pretrain_step(
                    model,
                    optimizer,
                    latents,
                    config.mae.mask_ratio,
                    config.mae.mask_patch_size,
                )
                metrics = {"reconstruction": loss}
            ema.update(model)
            step += 1
            if step % 100 == 0 or step == target_steps:
                print({"step": step, "loss": float(loss), **{k: float(v) for k, v in metrics.items()}})
            if step % 1000 == 0:
                write_checkpoint(step)
            if step >= target_steps:
                break
    write_checkpoint(step)
    print(f"Saved latent-MAE checkpoint to {output}")


if __name__ == "__main__":
    main()
