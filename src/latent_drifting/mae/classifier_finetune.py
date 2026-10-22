"""Linear-head-assisted MAE fine-tuning from Appendix A.3."""

from __future__ import annotations

import torch
from torch import nn

from latent_drifting.mae.model import LatentMAE, masked_reconstruction_loss


def classifier_mix_weight(step: int, warmup_steps: int = 1000, maximum: float = 0.1) -> float:
    if step < 0 or warmup_steps < 1 or not 0 <= maximum <= 1:
        raise ValueError("Invalid classifier fine-tuning schedule")
    return maximum * min(step / warmup_steps, 1.0)


def finetune_step(
    model: LatentMAE,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    warmup_steps: int = 1000,
    classifier_weight: float = 0.1,
    mask_ratio: float = 0.5,
    mask_patch_size: int = 2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if model.classifier is None:
        raise RuntimeError("Classifier fine-tuning requires LatentMAE(num_classes=...) ")
    model.train()
    output = model(inputs, mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
    reconstruction_loss = masked_reconstruction_loss(output, inputs)
    logits = model.classifier(output.features[-1].mean(dim=(2, 3)))
    classification_loss = nn.functional.cross_entropy(logits, labels)
    weight = classifier_mix_weight(step, warmup_steps, classifier_weight)
    loss = (1 - weight) * reconstruction_loss + weight * classification_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach(), {
        "reconstruction": reconstruction_loss.detach(),
        "classification": classification_loss.detach(),
        "classifier_weight": torch.tensor(weight, device=loss.device),
    }
