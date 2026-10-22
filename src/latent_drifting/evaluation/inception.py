"""Optional torchvision Inception-v3 feature and class-probability extractor."""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
        except ImportError as error:
            raise RuntimeError("Install latent-drifting[metrics] for Inception metrics") from error
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        self.model = inception_v3(weights=weights, transform_input=False, aux_logits=pretrained)
        self.model.eval()
        self._captured_features: torch.Tensor | None = None
        self.model.fc.register_forward_pre_hook(self._capture_features)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.requires_grad_(False)

    def _capture_features(self, module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        self._captured_features = args[0]

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        if images.shape[1] != 3:
            raise ValueError("Inception evaluation requires RGB images")
        images = images.float().add(1).div(2).clamp(0, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        logits = self.model((images - self.mean) / self.std)
        if not isinstance(logits, torch.Tensor) or self._captured_features is None:
            raise RuntimeError("Unexpected torchvision Inception output")
        return self._captured_features, logits


def inception_score(logits: torch.Tensor, splits: int = 10) -> tuple[float, float]:
    if logits.ndim != 2 or splits < 1:
        raise ValueError("logits must be [samples, classes] and splits positive")
    probabilities = logits.float().softmax(dim=-1)
    parts = probabilities.tensor_split(splits, dim=0)
    scores = []
    for part in parts:
        if part.shape[0] == 0:
            continue
        marginal = part.mean(dim=0, keepdim=True)
        divergence = part * (part.clamp_min(1e-16).log() - marginal.clamp_min(1e-16).log())
        scores.append(divergence.sum(dim=-1).mean().exp())
    values = torch.stack(scores)
    return float(values.mean()), float(values.std(unbiased=False))


class InceptionImageTransform:
    """Deterministically resize/crop real images to the generator's 256px protocol."""

    def __init__(self, resize_shorter: int = 256, crop_size: int = 256) -> None:
        self.resize_shorter = resize_shorter
        self.crop_size = crop_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.resize_shorter / min(width, height)
        resized = (round(width * scale), round(height * scale))
        image = image.resize(resized, Image.Resampling.BILINEAR)
        width, height = image.size
        left = max((width - self.crop_size) // 2, 0)
        top = max((height - self.crop_size) // 2, 0)
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(values.copy()).permute(2, 0, 1).contiguous()
