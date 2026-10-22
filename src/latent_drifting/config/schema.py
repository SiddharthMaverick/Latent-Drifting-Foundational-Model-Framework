"""Versioned configuration dataclasses and cross-section validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "drifting-experiment"
    seed: int = 0
    output_dir: str = "runs/default"
    config_version: int = 1


@dataclass(slots=True)
class DataConfig:
    root: str = "data/imagenet"
    num_classes: int = 1000
    image_size: int = 256
    latent_cache: str = "data/latent-cache"
    workers: int = 8
    pin_memory: bool = True
    random_resized_crop_scale: tuple[float, float] = (0.08, 1.0)


@dataclass(slots=True)
class VAEConfig:
    backend: str = "diffusers"
    model_id: str = "stabilityai/sd-vae-ft-mse"
    revision: str | None = None
    scale_factor: float = 0.18215
    dtype: str = "float32"


@dataclass(slots=True)
class MAEConfig:
    input_channels: int = 4
    base_width: int = 256
    blocks_per_stage: tuple[int, int, int, int] = (3, 4, 6, 3)
    mask_ratio: float = 0.5
    mask_patch_size: int = 2
    pretrain_epochs: int = 192
    finetune_steps: int = 3000
    classifier_warmup_steps: int = 1000
    classifier_weight: float = 0.1
    learning_rate: float = 4e-3
    ema_decay: float = 0.9995


@dataclass(slots=True)
class GeneratorConfig:
    variant: str = "B/2"
    input_channels: int = 4
    image_size: int = 32
    patch_size: int = 2
    hidden_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    num_classes: int = 1000
    register_tokens: int = 16
    style_tokens: int = 32
    style_codebook_size: int = 64
    use_style_tokens: bool = True
    dropout: float = 0.0


@dataclass(slots=True)
class DriftConfig:
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.2)
    distance: str = "l2"
    chunk_size: int = 1024
    normalize_features: bool = True
    normalize_drift: bool = True
    normalization_momentum: float = 0.99
    eps: float = 1e-6
    compute_dtype: str = "float32"


@dataclass(slots=True)
class CFGConfig:
    alpha_min: float = 1.0
    alpha_max: float = 4.0
    sampler: str = "power"
    power: float = 3.0
    num_unconditional: int = 32


@dataclass(slots=True)
class BatchConfig:
    classes: int = 128
    positives: int = 64
    negatives: int = 64
    local_size: int = 64
    accumulation: int = 1
    class_queue_size: int = 128
    unconditional_queue_size: int = 1000
    queue_push_size: int = 64


@dataclass(slots=True)
class OptimizationConfig:
    optimizer: str = "adamw"
    learning_rate: float = 4e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    total_steps: int = 200000
    clip_norm: float = 2.0
    precision: str = "bf16"
    ema_decays: tuple[float, ...] = (0.999, 0.9995, 0.9998, 0.9999)


@dataclass(slots=True)
class DistributedConfig:
    backend: str = "nccl"
    strategy: str = "ddp"
    class_grouping: bool = True
    find_unused_parameters: bool = False


@dataclass(slots=True)
class EvaluationConfig:
    num_images: int = 50000
    batch_size: int = 64
    fid_stats: str | None = None
    nearest_neighbors: int = 5
    sample_seed: int = 1234


@dataclass(slots=True)
class DeploymentConfig:
    format: str = "pytorch"
    opset: int = 18
    precision: str = "fp32"
    max_batch: int = 64


@dataclass(slots=True)
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    mae: MAEConfig = field(default_factory=MAEConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    cfg: CFGConfig = field(default_factory=CFGConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def validate(self) -> None:
        if self.experiment.config_version != 1:
            raise ValueError(f"Unsupported config version: {self.experiment.config_version}")
        if self.data.num_classes < 1 or self.generator.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if self.batch.classes > self.data.num_classes:
            raise ValueError("batch.classes cannot exceed data.num_classes")
        if self.generator.num_classes != self.data.num_classes:
            raise ValueError("data.num_classes and generator.num_classes must match")
        if self.generator.image_size % self.generator.patch_size:
            raise ValueError("generator.image_size must be divisible by patch_size")
        if self.generator.hidden_dim % self.generator.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.generator.hidden_dim // self.generator.num_heads % 4:
            raise ValueError("head dimension must be divisible by four for 2D RoPE")
        if not 0.0 <= self.mae.mask_ratio <= 1.0:
            raise ValueError("mae.mask_ratio must be in [0, 1]")
        if self.cfg.alpha_min < 1 or self.cfg.alpha_max < self.cfg.alpha_min:
            raise ValueError("CFG alpha bounds must satisfy 1 <= alpha_min <= alpha_max")
        if self.cfg.sampler not in {"power", "uniform", "fixed_one"}:
            raise ValueError("cfg.sampler must be power, uniform, or fixed_one")
        if not self.drift.temperatures or any(t <= 0 for t in self.drift.temperatures):
            raise ValueError("drift.temperatures must contain positive values")
        if self.drift.chunk_size < 1 or self.batch.accumulation < 1:
            raise ValueError("chunk_size and gradient accumulation must be positive")
        if self.optimization.total_steps < 1 or self.optimization.learning_rate <= 0:
            raise ValueError("optimization steps and learning rate must be positive")
        if self.batch.classes < 1 or self.batch.positives < 1 or self.batch.negatives < 2:
            raise ValueError("batch classes/positives must be positive and negatives >= 2")
        if self.generator.input_channels not in {3, 4}:
            raise ValueError("generator.input_channels must be 3 (pixel) or 4 (latent)")
        if self.cfg.num_unconditional < 0:
            raise ValueError("cfg.num_unconditional cannot be negative")
        if self.cfg.num_unconditional == 0 and self.cfg.alpha_max > 1:
            raise ValueError("CFG strengths above one require unconditional negative samples")
        if self.vae.scale_factor <= 0:
            raise ValueError("vae.scale_factor must be positive")
        if self.generator.input_channels == 3 and self.generator.image_size != 256:
            raise ValueError("pixel-space generation uses a 256x256 input")

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            return value

        return plain(asdict(self))

    def output_path(self) -> Path:
        return Path(self.experiment.output_dir)
