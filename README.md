# Latent-Drifting-Foundational-Model-Framework
Based on: "Generative Modeling via Drifting" (Deng et al., 2026)
Scope: Scaling from single-file prototype to distributed enterprise-grade infrastructure.

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the paper-grounded full-scale implementation plan.

## Package status

This repository contains an installable `latent-drifting` package with the mathematical drift core, class-conditional DiT, latent-MAE feature encoder, latent cache/queues, grouped training loop, checkpoint/EMA support, evaluation utilities, and generator-only export/runtime APIs.

The reference path is latent ImageNet generation: 256px RGB images are encoded into 32x32x4 SD-VAE latents, trained with a frozen multi-scale latent-MAE encoder, and decoded after one generator call. Optional pixel DiT variants operate directly on 256x256 RGB tensors. Large-scale ImageNet training requires an appropriately provisioned accelerator cluster and tokenizer/data access.

## Install

Use Python 3.10 or later and select the PyTorch build for the machine's OS/GPU. Then install the package:

```powershell
python -m pip install torch
python -m pip install -e .
```

Optional backends:

```powershell
python -m pip install -e ".[vae,metrics,export]"
```

The current workspace uses Python 3.15. For that interpreter, install PyTorch and torchvision from the official PyTorch wheel index appropriate to the host (CPU, CUDA, ROCm, or XPU) before installing this package; Python 3.15 wheels are not currently published on default PyPI, and `torch.compile` is not available on that interpreter. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and [PyTorch 2.14 release notes](https://pytorch.org/blog/pytorch-2-14-release-blog/).

## Workflow

1. Edit a versioned YAML config under `configs/`.
2. Cache VAE latents: `latent-drifting cache-latents --config configs/ablation_b2.yaml`.
3. Pretrain latent-MAE: `latent-drifting train-mae --config configs/ablation_b2.yaml`.
4. Train DiT with DDP: `torchrun --nproc_per_node=8 -m latent_drifting.cli.train_generator --config configs/ablation_b2.yaml --mae-checkpoint runs/ablation-b2/latent-mae.pt`.
5. Export the EMA generator only: `latent-drifting export --config configs/ablation_b2.yaml --checkpoint runs/ablation-b2/last.pt --ema-decay 0.9999 --output artifacts/drifting.pt`.
6. Sample: `latent-drifting sample --artifact artifacts/drifting.pt --count 16 --class-id 207`.
7. Compute reference Inception statistics: `latent-drifting compute-real-stats --root data/imagenet/val --count 50000 --output evaluation/imagenet-val.npz`.
8. Evaluate a fixed-seed sample set: `latent-drifting evaluate --artifact artifacts/drifting.pt --real-stats evaluation/imagenet-val.npz --count 50000 --output evaluation/run-001`.

For pixel-space training, use `configs/production_pixel_b16.yaml`, skip latent caching, and run `python -m latent_drifting.cli.train_mae --config configs/production_pixel_b16.yaml --pixel` before the same generator command.

For module invocation without installing the console script, use `python -m latent_drifting.cli.<command>`.

## Reproduction notes

- `data.num_classes` defaults to ImageNet's 1,000 classes. `batch.classes` is the per-step class allocation, not the dataset class count.
- Included configs encode the paper's B/2 ablation, B/2 production, and L/2 production regimes. Published FID values are reproduction targets, not test assertions.
- The training loop accepts cached positive latents and samples unconditional negatives from the cache. `SampleQueue` is also available for queue-driven data ingestion.
- Distributed class grouping partitions class IDs by rank. DDP is the end-to-end multi-GPU training path. The FSDP wrapper is only a lower-level integration hook; this trainer intentionally rejects FSDP until sharded EMA and full-state checkpoint handling are implemented.
- MAE resume restores model, optimizer, EMA, and RNG state; it does not restore the exact shuffled DataLoader position within an epoch. Generator resume also restores sampler epoch/offset and per-rank RNG state.
- Evaluation requires an independently prepared real-image stats file. Use a held-out validation folder for `compute-real-stats`; do not compute reference statistics from the training split.
- The code has not been executed in this workspace because its selected Python environment has no PyTorch installation; install the appropriate PyTorch build before running commands.
