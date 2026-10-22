"""Generate a fixed-seed sample set and compute FID/IS/nearest-neighbor diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from latent_drifting.cli.common import resolve_device, torch_dtype
from latent_drifting.deployment.artifact import DeploymentArtifact
from latent_drifting.evaluation.fid import StreamingStats, frechet_distance, load_stats, save_stats
from latent_drifting.evaluation.inception import InceptionFeatureExtractor, inception_score
from latent_drifting.evaluation.memorization import audit_memorization
from latent_drifting.evaluation.sampling import Sampler
from latent_drifting.utils.reproducibility import sha256_file
from latent_drifting.vae.diffusers_backend import DiffusersAutoencoder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--real-stats", required=True)
    parser.add_argument("--output", default="evaluation")
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vae-model", default=None)
    parser.add_argument("--training-features", default=None, help="Optional tensor file for nearest-neighbor audit")
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
    extractor = InceptionFeatureExtractor().to(device).eval()
    feature_stats = StreamingStats()
    logits_parts = []
    generated_features = []
    for start in range(0, args.count, args.batch_size):
        batch_count = min(args.batch_size, args.count - start)
        labels = torch.arange(start, start + batch_count, dtype=torch.long) % generator.config.num_classes
        images = sampler.generate(labels, args.alpha, args.seed + start)
        features, logits = extractor(images)
        feature_stats.update(features)
        if args.training_features:
            generated_features.append(features.cpu())
        logits_parts.append(logits.cpu())
    generated_stats = feature_stats.finalize()
    real_stats = load_stats(args.real_stats)
    fid = frechet_distance(real_stats.mean, real_stats.covariance, generated_stats.mean, generated_stats.covariance)
    score, score_std = inception_score(torch.cat(logits_parts))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_stats(generated_stats, str(output / "generated-stats.npz"))
    results = {"count": args.count, "fid": fid, "inception_score": score, "inception_score_std": score_std,
               "alpha": args.alpha, "seed": args.seed}
    if args.training_features:
        reference = torch.load(args.training_features, map_location="cpu", weights_only=True)
        if isinstance(reference, dict):
            reference = reference.get("features")
        if not isinstance(reference, torch.Tensor):
            raise TypeError("Training feature file must contain a tensor or a {'features': tensor} mapping")
        results["memorization"] = audit_memorization(torch.cat(generated_features), reference)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    manifest = {
        "artifact": str(Path(args.artifact).resolve()),
        "artifact_sha256": sha256_file(args.artifact),
        "real_stats": str(Path(args.real_stats).resolve()),
        "real_stats_sha256": sha256_file(args.real_stats),
        "sample_count": args.count,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "seed": args.seed,
        "generator_config": vars(generator.config) if hasattr(generator.config, "__dict__") else {
            field: getattr(generator.config, field) for field in generator.config.__dataclass_fields__
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(results)


if __name__ == "__main__":
    main()
