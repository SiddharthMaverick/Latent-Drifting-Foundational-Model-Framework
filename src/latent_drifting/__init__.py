"""Training and inference components for latent drifting generative models."""

from latent_drifting.config.schema import AppConfig
from latent_drifting.drifting.field import DriftingField
from latent_drifting.drifting.loss import DriftingLoss
from latent_drifting.models.generator import DriftingGenerator, GeneratorConfig

__version__ = "0.1.0"

from latent_drifting.deployment.artifact import DeploymentArtifact
from latent_drifting.evaluation.sampling import Sampler
from latent_drifting.features.hooks import FeatureEncoder
from latent_drifting.training.trainer import DriftingTrainer

__all__ = [
    "AppConfig",
    "DeploymentArtifact",
    "DriftingField",
    "DriftingGenerator",
    "DriftingLoss",
    "DriftingTrainer",
    "FeatureEncoder",
    "GeneratorConfig",
    "Sampler",
]
