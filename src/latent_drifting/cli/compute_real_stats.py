"""Compute and save Inception activation statistics for a reference image folder."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from latent_drifting.cli.common import resolve_device
from latent_drifting.data.image_dataset import ImageFolderDataset
from latent_drifting.evaluation.fid import StreamingStats, save_stats
from latent_drifting.evaluation.inception import InceptionFeatureExtractor, InceptionImageTransform


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    dataset = ImageFolderDataset(args.root, InceptionImageTransform())
    if args.count > len(dataset):
        raise ValueError(f"Requested {args.count} images but dataset contains {len(dataset)}")
    loader = DataLoader(
        Subset(dataset, range(args.count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = InceptionFeatureExtractor().to(device).eval()
    stats = StreamingStats()
    for images, _ in loader:
        features, _ = model(images.to(device, non_blocking=True))
        stats.update(features)
    save_stats(stats.finalize(), args.output)
    print(f"Saved Inception statistics for {args.count} images to {args.output}")


if __name__ == "__main__":
    main()
