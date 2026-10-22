# Model card: Latent Drifting Framework

## Model family

One-pass, class-conditioned generators trained by minimizing a kernelized drifting field. The repository provides DiT-B/2 and DiT-L/2 latent configurations and a DiT-B/16 pixel configuration.

## Intended use

Research reproduction and experimentation on class-conditional image generation. The package also exposes the distribution-drift core for adapting to other feature spaces.

## Training data

The supplied configurations target ImageNet-1k. Dataset files and pretrained VAE/feature weights are not included. Users must provide the data and confirm that their use is permitted under the relevant licenses and terms.

## Limitations

- No pretrained generator checkpoint is shipped.
- The repository implementation has not yet produced or independently verified the paper's reported FID/IS values.
- Quality, class fidelity, and memorization behavior depend on the selected tokenizer, feature encoder, preprocessing, training scale, and evaluation implementation.
- TensorRT, ONNX, and distributed execution require compatible external runtimes and hardware.

## Evaluation

Use fixed evaluation manifests, 50K samples, a documented Inception implementation, nearest-neighbor audits, and class-conditional qualitative inspection. Record the exact config, package revision, VAE identifier, feature checkpoint hash, and seeds with each report.

