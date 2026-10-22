"""Minimal, versioned inference artifact containing only generator weights."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from latent_drifting import __version__
from latent_drifting.config.schema import GeneratorConfig
from latent_drifting.models.generator import DriftingGenerator


class DeploymentArtifact:
    FORMAT_VERSION = 1

    def __init__(
        self,
        generator_config: GeneratorConfig,
        generator_state: dict[str, torch.Tensor],
        vae_reference: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.generator_config = generator_config
        self.generator_state = generator_state
        self.vae_reference = vae_reference
        self.metadata = metadata or {}

    @classmethod
    def from_generator(
        cls,
        generator: DriftingGenerator,
        vae_reference: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeploymentArtifact:
        state = {name: value.detach().cpu().clone() for name, value in generator.state_dict().items()}
        return cls(generator.config, state, vae_reference, metadata)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "package_version": __version__,
            "generator_config": asdict(self.generator_config),
            "generator": self.generator_state,
            "vae_reference": self.vae_reference,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> tuple[DeploymentArtifact, DriftingGenerator]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported deployment artifact format")
        config = GeneratorConfig(**payload["generator_config"])
        artifact = cls(config, payload["generator"], payload.get("vae_reference"), payload.get("metadata"))
        generator = DriftingGenerator(config).to(device)
        generator.load_state_dict(artifact.generator_state)
        generator.eval()
        return artifact, generator
