"""Reference single-process and distributed training loop for grouped classes."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from latent_drifting.drifting.loss import DriftingLoss
from latent_drifting.training.ema import MultiEMA


@dataclass(slots=True)
class GroupedDriftingBatch:
    """Class-major training inputs: ``[classes, samples_per_class, C, H, W]``."""

    noise: torch.Tensor
    labels: torch.Tensor
    positives: torch.Tensor
    alphas: torch.Tensor
    unconditional: torch.Tensor | None = None

    def validate(self) -> None:
        if self.noise.ndim != 5 or self.positives.ndim != 5:
            raise ValueError("noise and positives must have [classes, samples, C, H, W] shape")
        if self.labels.shape != (self.noise.shape[0],):
            raise ValueError("labels must have one class id per grouped class")
        if self.alphas.shape != self.labels.shape:
            raise ValueError("alphas must have one guidance value per grouped class")
        if self.noise.shape[0] != self.positives.shape[0]:
            raise ValueError("noise and positive class counts differ")
        if self.noise.shape[2:] != self.positives.shape[2:]:
            raise ValueError("noise and positive sample shapes differ")
        if self.unconditional is not None:
            if self.unconditional.ndim != 5 or self.unconditional.shape[0] != self.noise.shape[0]:
                raise ValueError("unconditional samples must be [classes, samples, C, H, W]")
            if self.unconditional.shape[2:] != self.noise.shape[2:]:
                raise ValueError("unconditional sample shape differs from generator input")

    def to(self, device: torch.device | str) -> GroupedDriftingBatch:
        return GroupedDriftingBatch(
            self.noise.to(device, non_blocking=True),
            self.labels.to(device, non_blocking=True),
            self.positives.to(device, non_blocking=True),
            self.alphas.to(device, non_blocking=True),
            None if self.unconditional is None else self.unconditional.to(device, non_blocking=True),
        )


class DriftingTrainer:
    def __init__(
        self,
        generator: nn.Module,
        feature_encoder: nn.Module,
        drifting_loss: DriftingLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        precision: str = "bf16",
        clip_norm: float = 2.0,
        ema: MultiEMA | None = None,
    ) -> None:
        self.generator = generator.to(device)
        self.raw_generator = getattr(self.generator, "module", self.generator)
        self.feature_encoder = feature_encoder.to(device)
        self.feature_encoder.requires_grad_(False)
        self.feature_encoder.eval()
        self.drifting_loss = drifting_loss
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.precision = precision
        self.clip_norm = clip_norm
        self.ema = ema
        self.step = 0
        self.micro_step = 0
        self.accumulation_count = 0
        self._scaler = torch.cuda.amp.GradScaler(
            enabled=self.device.type == "cuda" and precision == "fp16"
        )

    def _autocast(self):
        enabled = self.device.type == "cuda" and self.precision in {"fp16", "bf16"}
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def train_step(
        self,
        batch: GroupedDriftingBatch,
        accumulation_steps: int = 1,
    ) -> dict[str, float]:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        if self.accumulation_count >= accumulation_steps:
            raise RuntimeError("Accumulation state exceeds requested accumulation_steps")
        batch.validate()
        batch = batch.to(self.device)
        classes, negatives_per_class = batch.noise.shape[:2]
        positives_per_class = batch.positives.shape[1]
        flat_noise = batch.noise.flatten(0, 1)
        flat_labels = batch.labels.repeat_interleave(negatives_per_class)
        flat_alpha = batch.alphas.repeat_interleave(negatives_per_class)
        self.generator.train()
        self.feature_encoder.eval()
        if self.accumulation_count == 0:
            self.optimizer.zero_grad(set_to_none=True)
        optimizer_step = self.accumulation_count + 1 == accumulation_steps
        sync_context = (
            self.generator.no_sync()
            if not optimizer_step and hasattr(self.generator, "no_sync")
            else contextlib.nullcontext()
        )
        with sync_context:
            with self._autocast():
                generated = self.generator(flat_noise, flat_labels, flat_alpha)
                generated_features = self.feature_encoder(generated)
                with torch.no_grad():
                    positive_features = self.feature_encoder(batch.positives.flatten(0, 1))
                    unconditional_features = (
                        None
                        if batch.unconditional is None
                        else self.feature_encoder(batch.unconditional.flatten(0, 1))
                    )
                class_losses: list[torch.Tensor] = []
                metric_sums: dict[str, float] = {}
                for class_index in range(classes):
                    generated_start = class_index * negatives_per_class
                    positive_start = class_index * positives_per_class
                    generated_group = {
                        key: value[generated_start : generated_start + negatives_per_class]
                        for key, value in generated_features.items()
                    }
                    positive_group = {
                        key: value[positive_start : positive_start + positives_per_class]
                        for key, value in positive_features.items()
                    }
                    unconditional_group = None
                    if unconditional_features is not None and batch.unconditional is not None:
                        count = batch.unconditional.shape[1]
                        start = class_index * count
                        unconditional_group = {
                            key: value[start : start + count]
                            for key, value in unconditional_features.items()
                        }
                    output = self.drifting_loss(
                        generated_group,
                        positive_group,
                        unconditional_group,
                        batch.alphas[class_index],
                        negative_is_generated=True,
                    )
                    class_losses.append(output.loss)
                    for name, metric in output.metrics.items():
                        metric_sums[name] = metric_sums.get(name, 0.0) + float(metric.float().mean()) / classes
                loss = torch.stack(class_losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Drifting objective produced a non-finite loss")
            self._scaler.scale(loss / accumulation_steps).backward()
        self.micro_step += 1
        self.accumulation_count += 1
        grad_norm_value = 0.0
        if optimizer_step:
            self._scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.clip_norm, error_if_nonfinite=True
            )
            grad_norm_value = float(grad_norm.detach())
            self._scaler.step(self.optimizer)
            self._scaler.update()
            if self.ema is not None:
                self.ema.update(self.raw_generator)
            self.step += 1
            self.accumulation_count = 0
        metric_sums["loss"] = float(loss.detach())
        metric_sums["grad_norm"] = grad_norm_value
        metric_sums["generated_per_step"] = float(classes * negatives_per_class)
        metric_sums["optimizer_step"] = float(optimizer_step)
        return metric_sums

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "micro_step": self.micro_step,
            "accumulation_count": self.accumulation_count,
            "drifting_loss": self.drifting_loss.state_dict(),
            "scaler": self._scaler.state_dict(),
            "ema": None if self.ema is None else self.ema.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step = int(state["step"])
        self.micro_step = int(state.get("micro_step", self.step))
        self.accumulation_count = int(state.get("accumulation_count", 0))
        self.drifting_loss.load_state_dict(state["drifting_loss"])
        self._scaler.load_state_dict(state["scaler"])
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])
