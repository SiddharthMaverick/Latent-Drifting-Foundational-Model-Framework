# Latent Drifting Foundation Model Framework

## Full-scale implementation plan

This document translates *Generative Modeling via Drifting* (Deng et al., arXiv:2602.04770v2) and the repository blueprint into an implementable, testable Python/PyTorch package.

## 1. Scope and non-negotiable behavior

The package will support:

- latent-space ImageNet generation at 256x256 using a frozen SD-VAE;
- optional pixel-space generation at 256x256;
- class-conditional generation with training-time CFG conditioning;
- one neural-network evaluation at inference, followed by VAE decoding for latent models;
- single-GPU smoke tests, multi-GPU distributed training, checkpoint recovery, evaluation, and export;
- reusable drifting-field and feature-space components for non-image data.

The implementation must preserve the paper’s central invariant: the drift is anti-symmetric when positive and negative distributions are exchanged, and the training target is detached:

```text
V = V_positive - V_negative
target = stop_gradient(features(x) + normalized(V))
loss = MSE(features(x), target)
```

No iterative ODE/SDE solver, denoising loop, or feature encoder is part of the inference artifact.

## 2. Reconciled requirements

The paper is authoritative where it conflicts with the initial blueprint.

| Area | Paper-backed production behavior | Implementation decision |
|---|---|---|
| Dataset | ImageNet 256x256, 1,000 classes | Make `num_classes` configurable; default to 1,000. Never hard-code 128. |
| Latent shape | 32x32x4 SD-VAE latent | Validate layout and VAE scaling at dataset-cache time. |
| Generator | DiT-B/2 or DiT-L/2; 256 tokens | Implement B/2 first, then L/2 from the same module. |
| Conditioning | class `c` and CFG strength `alpha` | Support labels, null/unconditional labels, and continuous alpha in every forward. |
| Feature encoder | ResNet-style latent-MAE; frozen during generator training | Implement latent-MAE first; keep pixel encoders behind the same interface. |
| Feature pyramid | Per-location, global, 2x2 and 4x4 pooled mean/std, input x² statistics | Represent every descriptor as a typed feature group with shared normalization. |
| Queues | Per-class queue size 128; global unconditional queue size 1,000; push 64 | Make queue sizes and push count configurable and checkpointable. |
| Drift | Multi-temperature bilateral/dual-softmax kernel | Implement numerically stable log-softmax and explicit self-exclusion. |
| CFG | alpha in [1,4], sampled by configured distribution; paper reports alpha^-3 and alpha^-5 variants | Provide exact samplers and record the selected schedule in checkpoints. |
| Scale | Effective batch 8,192 in production | Separate local batch, class-group batch, and effective batch in configuration/logging. |
| Results | Reported FID 1.75 for B/2 and 1.54 for L/2 in the paper’s protocol | Treat these as reproduction targets, not acceptance guarantees. |

The blueprint’s `N_c=128` should be interpreted as a batch allocation or class-grouping parameter, not the number of ImageNet classes.

## 3. Package architecture

```text
latent_drifting/
  cli/
    cache_latents.py
    train_mae.py
    train_generator.py
    evaluate.py
    sample.py
    export.py
  config/
    schema.py
    loaders.py
  data/
    image_dataset.py
    transforms.py
    vae_cache.py
    grouped_batch_sampler.py
    sample_queue.py
    distributed_queue.py
  vae/
    interface.py
    diffusers_backend.py
  mae/
    resnet_encoder.py
    decoder.py
    masking.py
    model.py
    pretrain.py
    classifier_finetune.py
  features/
    pyramid.py
    descriptors.py
    normalization.py
    hooks.py
  models/
    patchify.py
    rope.py
    norm.py
    conditioning.py
    dit_block.py
    generator.py
  drifting/
    distances.py
    bilateral_kernel.py
    cfg.py
    field.py
    loss.py
  training/
    trainer.py
    distributed.py
    ema.py
    checkpoint.py
    metrics.py
  evaluation/
    sampling.py
    fid.py
    inception.py
    nearest_neighbors.py
    memorization.py
  deployment/
    artifact.py
    export_torchscript.py
    export_onnx.py
    export_tensorrt.py
    runtime.py
  utils/
    reproducibility.py
    logging.py
    profiling.py
```

The public API should expose only stable interfaces such as `DriftingGenerator`, `FeatureEncoder`, `DriftingField`, `DriftingLoss`, `Trainer`, `Sampler`, and `DeploymentArtifact`. Internal tensor layouts and distributed implementation details remain private.

## 4. Core tensor contracts

Use NCHW internally for convolutional modules and `[batch, descriptors, channels]` for the feature-space kernel.

- latent input/output: `[B, 4, 32, 32]`;
- pixel input/output: `[B, 3, 256, 256]`;
- latent DiT tokens: `[B, 256, hidden_dim]`;
- class labels: `[B]`, integer class IDs;
- alpha: `[B]`, floating-point CFG strengths;
- feature descriptor group `j`: `[B, M_j, C_j]`, where `M_j` is the number of spatial/statistical locations;
- generated negatives and positives must retain class/group membership throughout batching.

Every public tensor-producing module must validate shape, dtype, device, and finite values in debug mode. Mixed precision is allowed for model activations, but pairwise distance reductions, normalization statistics, and softmax logits must have a configurable higher-precision path.

## 5. Implementation sequence

### Phase 0 — Foundation and reproducibility (week 1)

- create `pyproject.toml`, package metadata, lint/type/test configuration, and versioned YAML schemas;
- implement deterministic seed handling, structured JSON logging, checkpoint metadata, and experiment manifests;
- add CPU/GPU smoke fixtures with tiny models and synthetic distributions;
- define the artifact format and compatibility policy.

Exit criteria: a clean install can run a synthetic end-to-end train, resume, sample, and evaluate cycle.

### Phase 1 — Data and tokenizer pipeline (weeks 1–2)

- implement ImageNet-style dataset indexing and class-balanced grouped sampling;
- integrate the frozen SD-VAE backend and pixel-space random resized crop before encoding;
- cache latents in sharded, checksummed files with class labels and preprocessing metadata;
- implement per-class/global FIFO queues with distributed-safe checkpoint/restore;
- validate latent statistics against the selected VAE implementation.

Exit criteria: cache validation reports exact shape, class coverage, finite values, reproducible samples, and no train/validation leakage.

### Phase 2 — Latent-MAE feature encoder (weeks 2–4)

- implement the ResNet-34 basic-block encoder with GroupNorm and stages `[3,4,6,3]`;
- implement zeroed 2x2 masking at 50% and the U-Net-style reconstruction decoder;
- add masked reconstruction pretraining with EMA and checkpoint resume;
- add classifier fine-tuning with the paper’s reconstruction/classification warmup;
- expose frozen multi-stage hooks and descriptor extraction;
- add optional MoCo/SimCLR and pixel-MAE adapters later, without coupling them to the generator.

Exit criteria: reconstruction loss decreases, feature shapes match all four resolutions, classifier fine-tuning is reproducible, and encoder gradients are absent during generator training.

### Phase 3 — Mathematical core (weeks 3–5)

- implement pairwise L2 distances with chunking to avoid materializing unbounded matrices;
- implement self-exclusion for generated samples used as negatives;
- implement row/column log-softmax, geometric combination, positive/negative weighting, and drift calculation;
- implement feature normalization `S_j`, drift normalization `lambda_j`, shared spatial scales, and multiple temperatures;
- implement CFG alpha sampling and the unconditional-negative weight derived from alpha;
- implement the detached fixed-point MSE loss.

Exit criteria: analytic and numerical tests verify zero self-drift, anti-symmetry, finite gradients, correct CFG weights, and equivalence between dense and chunked kernels.

### Phase 4 — Generator (weeks 4–6)

- implement patchify/unpatchify and DiT-B/2;
- implement RMSNorm, QK-Norm, SwiGLU, 2D RoPE, adaLN-Zero, 16 register tokens, and 32 optional random style tokens;
- implement class and alpha conditioning with checkpointed embedding metadata;
- add DiT-L/2 configuration only after B/2 parity is established;
- implement EMA copies and sampling with exactly one generator forward.

Exit criteria: parameter counts, token counts, output shapes, initialization behavior, and one-forward inference are verified against configuration snapshots.

### Phase 5 — Training engine and distributed scale (weeks 5–7)

- implement the seven-step paper loop: labels, alpha, noise, positives/unconditional samples, features, drift loss, optimizer update;
- implement gradient accumulation, AMP, clipping, AdamW, warmup, EMA, periodic evaluation, and fault-tolerant resume;
- implement DDP/FSDP modes, rank-local queues, class-group ownership, and explicit communication boundaries;
- record effective batch and sample counts separately from local batch sizes;
- add profiling for feature extraction, distance kernels, communication, and checkpoint I/O.

Exit criteria: single-node multi-GPU training produces matching loss/metric traces across restart and does not exceed configured memory limits.

### Phase 6 — Evaluation and deployment (weeks 7–8)

- generate fixed 50K evaluation sets with immutable seeds and manifests;
- implement FID, IS, class-conditional metrics, nearest-neighbor/memorization checks, and drift diagnostics;
- export generator-only weights plus VAE decoder reference/configuration;
- implement TorchScript/ONNX first, then TensorRT with an explicit fallback path;
- provide a runtime API that accepts noise, class, alpha, and seed and returns image tensors.

Exit criteria: exported and eager runtimes agree within numerical tolerance, inference uses one generator call, and the artifact contains no feature encoder, queue, or kernel state.

## 6. Configuration strategy

Use typed, versioned configs rather than unvalidated free-form YAML. Minimum config sections:

```yaml
experiment: {name, seed, output_dir}
data: {root, num_classes, image_size, latent_cache, workers}
vae: {backend, model_id, scale_factor, dtype}
mae: {base_width, blocks, mask_ratio, pretrain_epochs, finetune_steps}
generator: {variant, patch_size, hidden_dim, depth, registers, style_tokens}
drift: {temperatures, distance, chunk_size, normalize_features, normalize_drift}
cfg: {alpha_min, alpha_max, sampler, num_unconditional}
batch: {classes, positives, negatives, local_size, accumulation}
optimization: {optimizer, lr, betas, weight_decay, warmup, clip_norm}
distributed: {backend, strategy, class_grouping, precision}
evaluation: {num_images, fid_stats, nearest_neighbors, sample_seed}
deployment: {format, opset, precision, max_batch}
```

Configs must be saved verbatim with every checkpoint, along with git revision, package version, dataset manifest hash, VAE identifier, and feature-encoder checkpoint hash.

## 7. Testing and validation

### Unit tests

- patchify/unpatchify round trip;
- RoPE, RMSNorm, QK-Norm, adaLN-Zero, masking, descriptor shapes;
- queue FIFO order, no-replacement sampling, class isolation, and restore;
- CFG alpha distributions and unconditional weights;
- anti-symmetry and zero self-drift;
- dense/chunked kernel equivalence;
- normalization invariance under feature rescaling;
- detached target and generator-only gradient flow.

### Integration tests

- synthetic 2D bimodal convergence from three initializations;
- tiny latent-MAE pretraining and fine-tuning;
- tiny generator training with a frozen encoder;
- DDP two-rank queue and checkpoint restart;
- export/import parity and one-forward inference audit.

### Reproduction gates

1. Reproduce the synthetic distribution figures and decreasing drift norm.
2. Reproduce B/2 ablation trends before scaling.
3. Reproduce feature-encoder ablation ordering before long training.
4. Run 100-epoch B/2 baseline and publish FID/IS with exact manifests.
5. Run longer B/2, then L/2 only after profiling and numerical parity.

The paper’s FID values are benchmark references. They should not be encoded as brittle test assertions because FID depends on data preprocessing, VAE version, evaluation implementation, and hardware/numerical choices.

## 8. Operational safeguards

- Keep train, validation, and evaluation manifests immutable and hashed.
- Add NaN/Inf detection around distances, softmax, normalization, and gradients.
- Fail fast when queue warm-up is insufficient instead of silently sampling invalid classes.
- Record queue occupancy, drift norm, feature scales, kernel entropy, gradient norms, and per-class losses.
- Make all expensive pairwise operations chunkable and profile their peak memory.
- Never include training queues, MAE weights, or feature-extractor code in deployment exports.
- Maintain a model card documenting dataset provenance, license, intended use, limitations, and memorization checks.

## 9. Immediate next deliverables

1. Bootstrap package metadata and typed configuration schema.
2. Implement the mathematical core with toy tests before integrating ImageNet.
3. Implement the latent cache and queue interfaces.
4. Implement latent-MAE and descriptor extraction.
5. Implement DiT-B/2 and a tiny end-to-end training command.
6. Add distributed training only after the single-GPU reference path is numerically locked.

This ordering makes the implementation scientifically auditable: each scaling step depends on a smaller reference implementation whose behavior can be tested independently.

## References

- Deng, Li, Li, Du, He. *Generative Modeling via Drifting*. arXiv:2602.04770v2, 2026.
- Repository blueprint: `latent_drifting_foundation_model_framework_blueprint.txt`.
