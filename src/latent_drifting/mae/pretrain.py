"""One-step and epoch-level latent-MAE pretraining helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from latent_drifting.mae.model import LatentMAE, masked_reconstruction_loss


def pretrain_step(
    model: LatentMAE,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    mask_ratio: float = 0.5,
    mask_patch_size: int = 2,
    clip_norm: float = 2.0,
) -> torch.Tensor:
    model.train()
    output = model(inputs, mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
    loss = masked_reconstruction_loss(output, inputs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    return loss.detach()


def pretrain_epoch(
    model: LatentMAE,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    device: torch.device | str,
    mask_ratio: float = 0.5,
    mask_patch_size: int = 2,
    clip_norm: float = 2.0,
) -> float:
    total, count = 0.0, 0
    for batch in batches:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        loss = pretrain_step(
            model,
            optimizer,
            inputs.to(device, non_blocking=True),
            mask_ratio,
            mask_patch_size,
            clip_norm,
        )
        total += float(loss)
        count += 1
    if count == 0:
        raise RuntimeError("MAE pretraining loader yielded no batches")
    return total / count
