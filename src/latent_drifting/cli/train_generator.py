"""Train DiT with the feature-space drifting objective on cached latents."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_drifting.cli.common import resolve_device
from latent_drifting.config.loaders import load_config
from latent_drifting.data.grouped_batch_sampler import ClassGroupedBatchSampler
from latent_drifting.data.image_dataset import ImageFolderDataset
from latent_drifting.data.transforms import ImageTransform
from latent_drifting.data.vae_cache import LatentCacheDataset
from latent_drifting.drifting.cfg import sample_cfg_alpha
from latent_drifting.drifting.loss import DriftingLoss
from latent_drifting.features.hooks import FeatureEncoder
from latent_drifting.features.pixel import PixelPatchifier
from latent_drifting.mae.resnet_encoder import ResNetMAEEncoder
from latent_drifting.models.generator import DriftingGenerator
from latent_drifting.training.checkpoint import load_checkpoint, save_checkpoint
from latent_drifting.training.distributed import (
    all_reduce_metrics,
    class_indices_for_rank,
    initialize_distributed,
    wrap_distributed,
)
from latent_drifting.training.ema import MultiEMA
from latent_drifting.training.trainer import DriftingTrainer, GroupedDriftingBatch
from latent_drifting.utils.logging import create_logger, log_metrics
from latent_drifting.utils.reproducibility import capture_rng_state, restore_rng_state, seed_everything


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mae-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.distributed.strategy == "fsdp":
        raise NotImplementedError(
            "End-to-end FSDP training is not enabled: sharded EMA and full-state checkpoint "
            "handling must be integrated before using this trainer. Use distributed.strategy=ddp."
        )
    rank, world_size, local_rank = initialize_distributed(config.distributed.backend)
    device = resolve_device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank)
    seed_everything(config.experiment.seed)
    pixel_space = config.generator.input_channels == 3
    if pixel_space:
        dataset = ImageFolderDataset(
            config.data.root,
            ImageTransform(config.data.image_size, config.data.random_resized_crop_scale),
        )
        if len(dataset.class_to_idx) != config.data.num_classes:
            raise ValueError(
                f"Image dataset has {len(dataset.class_to_idx)} classes, expected {config.data.num_classes}"
            )
    else:
        dataset = LatentCacheDataset(args.cache or config.data.latent_cache)
    missing_classes = set(range(config.data.num_classes)) - set(dataset.indices_by_class)
    if missing_classes:
        preview = sorted(missing_classes)[:8]
        raise ValueError(f"Training source is missing configured classes; first missing ids: {preview}")
    if config.batch.classes % world_size:
        raise ValueError("batch.classes must be divisible by distributed world size")
    rank_class_ids = class_indices_for_rank(config.data.num_classes, rank, world_size)
    local_index_map = {
        class_id: dataset.indices_by_class[class_id]
        for class_id in rank_class_ids
        if class_id in dataset.indices_by_class
    }
    local_classes = config.batch.classes // world_size
    if local_classes > len(local_index_map):
        raise ValueError("This rank owns fewer classes than its configured local class batch")
    batch_sampler = ClassGroupedBatchSampler(
        local_index_map,
        local_classes,
        config.batch.positives,
        seed=config.experiment.seed + rank,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.data.workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.workers > 0,
    )
    generator = DriftingGenerator(config.generator).to(device)
    mae_checkpoint = torch.load(args.mae_checkpoint, map_location="cpu", weights_only=False)
    if bool(mae_checkpoint.get("pixel_space", False)) != pixel_space:
        raise ValueError("MAE checkpoint pixel/latent mode must match generator.input_channels")
    encoder_input_channels = 192 if pixel_space else config.mae.input_channels
    encoder_prefix = "model.encoder." if pixel_space else "encoder."
    encoder_state = {
        name.removeprefix(encoder_prefix): value
        for name, value in mae_checkpoint["model"].items()
        if name.startswith(encoder_prefix)
    }
    if not encoder_state:
        raise ValueError("MAE checkpoint contains no compatible encoder weights")
    encoder = ResNetMAEEncoder(
        encoder_input_channels,
        config.mae.base_width,
        config.mae.blocks_per_stage,
    )
    encoder.load_state_dict(encoder_state, strict=True)
    feature_encoder = FeatureEncoder(
        encoder,
        input_projection=PixelPatchifier() if pixel_space else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.optimization.learning_rate,
        betas=config.optimization.betas,
        weight_decay=config.optimization.weight_decay,
    )
    def warmup_scale(step: int) -> float:
        if config.optimization.warmup_steps <= 0:
            return 1.0
        return min((step + 1) / config.optimization.warmup_steps, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_scale)
    ema = MultiEMA(generator, config.optimization.ema_decays)
    seed_everything(config.experiment.seed + rank)
    objective = DriftingLoss(
        config.drift.temperatures,
        config.drift.chunk_size,
        config.drift.normalize_features,
        config.drift.normalize_drift,
        config.drift.normalization_momentum,
        config.drift.eps,
    )
    distributed_generator = wrap_distributed(
        generator,
        config.distributed.strategy,
        local_rank if device.type == "cuda" else None,
        config.distributed.find_unused_parameters,
    )
    trainer = DriftingTrainer(
        distributed_generator,
        feature_encoder,
        objective,
        optimizer,
        device,
        config.optimization.precision,
        config.optimization.clip_norm,
        ema,
    )
    resume_extra = {}
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            generator,
            optimizer,
            ema=ema,
            scaler=trainer._scaler,
            scheduler=scheduler,
            loss_object=objective,
            map_location=device,
        )
        trainer.step = int(payload["step"])
        resume_extra = payload.get("extra", {})
        if "trainer" in resume_extra:
            trainer.load_state_dict(resume_extra["trainer"])
        rank_rng_states = resume_extra.get("rank_rng_states")
        if rank_rng_states is not None and rank < len(rank_rng_states):
            restore_rng_state(rank_rng_states[rank])
        batch_sampler.set_epoch(int(resume_extra.get("sampler_epoch", 0)))
    output_dir = Path(args.output or config.experiment.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("latent_drifting.train", output_dir / "train.jsonl" if rank == 0 else None)
    target_steps = args.steps or config.optimization.total_steps
    data_iterator = iter(loader)
    batches_in_epoch = int(resume_extra.get("batches_in_epoch", 0))
    for _ in range(batches_in_epoch):
        try:
            next(data_iterator)
        except StopIteration as error:
            raise ValueError("Checkpoint batch offset exceeds the current epoch length") from error
    while trainer.step < target_steps:
        try:
            real, labels = next(data_iterator)
        except StopIteration:
            batch_sampler.set_epoch(batch_sampler.epoch + 1)
            batches_in_epoch = 0
            data_iterator = iter(loader)
            real, labels = next(data_iterator)
        batches_in_epoch += 1
        class_count = local_classes
        positive_count = config.batch.positives
        labels = labels.view(class_count, positive_count)
        class_ids = labels[:, 0]
        if torch.any(labels != class_ids[:, None]):
            raise RuntimeError("ClassGroupedBatchSampler returned a non-class-major batch")
        positives = real.view(class_count, positive_count, *real.shape[1:])
        negative_count = config.batch.negatives
        noise = torch.randn(
            class_count,
            negative_count,
            config.generator.input_channels,
            config.generator.image_size,
            config.generator.image_size,
        )
        unconditional = None
        if config.cfg.num_unconditional:
            count = config.cfg.num_unconditional
            examples = [dataset[random.randrange(len(dataset))][0] for _ in range(class_count * count)]
            unconditional = torch.stack(examples).view(class_count, count, *real.shape[1:])
        alphas = sample_cfg_alpha(
            class_count,
            config.cfg.alpha_min,
            config.cfg.alpha_max,
            config.cfg.power,
            config.cfg.sampler,
        )
        metrics = trainer.train_step(
            GroupedDriftingBatch(noise, class_ids, positives, alphas, unconditional),
            config.batch.accumulation,
        )
        if metrics["optimizer_step"]:
            scheduler.step()
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        if torch.distributed.is_initialized():
            metrics = all_reduce_metrics(metrics)
        metrics["global_generated_per_step"] = float(
            config.batch.classes * config.batch.negatives * config.batch.accumulation
        )
        if rank == 0 and metrics["optimizer_step"] and trainer.step % 20 == 0:
            log_metrics(logger, trainer.step, metrics)
        if metrics["optimizer_step"] and (trainer.step % 1000 == 0 or trainer.step == target_steps):
            local_rng_state = capture_rng_state()
            if torch.distributed.is_initialized():
                rank_rng_states = [None] * world_size
                torch.distributed.all_gather_object(rank_rng_states, local_rng_state)
            else:
                rank_rng_states = [local_rng_state]
            if rank == 0:
                save_checkpoint(
                    output_dir / f"checkpoint-{trainer.step:08d}.pt",
                    generator,
                    optimizer,
                    trainer.step,
                    config.to_dict(),
                    ema=ema,
                    scaler=trainer._scaler,
                    scheduler=scheduler,
                    loss_state=objective.state_dict(),
                    extra={
                        "trainer": trainer.state_dict(),
                        "sampler_epoch": batch_sampler.epoch,
                        "batches_in_epoch": batches_in_epoch,
                        "rank_rng_states": rank_rng_states,
                    },
                )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
    local_rng_state = capture_rng_state()
    if torch.distributed.is_initialized():
        rank_rng_states = [None] * world_size
        torch.distributed.all_gather_object(rank_rng_states, local_rng_state)
    else:
        rank_rng_states = [local_rng_state]
    if rank == 0:
        save_checkpoint(
            output_dir / "last.pt",
            generator,
            optimizer,
            trainer.step,
            config.to_dict(),
            ema=ema,
            scaler=trainer._scaler,
            scheduler=scheduler,
            loss_state=objective.state_dict(),
            extra={
                "trainer": trainer.state_dict(),
                "sampler_epoch": batch_sampler.epoch,
                "batches_in_epoch": batches_in_epoch,
                "rank_rng_states": rank_rng_states,
            },
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
