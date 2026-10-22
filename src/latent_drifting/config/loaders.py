"""Load and save validated versioned YAML configs."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar, get_origin, get_type_hints

import yaml

from latent_drifting.config.schema import (
    AppConfig,
    BatchConfig,
    CFGConfig,
    DataConfig,
    DeploymentConfig,
    DistributedConfig,
    DriftConfig,
    EvaluationConfig,
    ExperimentConfig,
    GeneratorConfig,
    MAEConfig,
    OptimizationConfig,
    VAEConfig,
)

T = TypeVar("T")

_SECTION_TYPES: dict[str, type[Any]] = {
    "experiment": ExperimentConfig,
    "data": DataConfig,
    "vae": VAEConfig,
    "mae": MAEConfig,
    "generator": GeneratorConfig,
    "drift": DriftConfig,
    "cfg": CFGConfig,
    "batch": BatchConfig,
    "optimization": OptimizationConfig,
    "distributed": DistributedConfig,
    "evaluation": EvaluationConfig,
    "deployment": DeploymentConfig,
}


def _coerce_section(section_name: str, values: Any, cls: type[T]) -> T:
    if not isinstance(values, dict):
        raise TypeError(f"Config section {section_name!r} must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {section_name}: {', '.join(sorted(unknown))}")
    tuple_fields = {
        name for name, annotation in get_type_hints(cls).items() if get_origin(annotation) is tuple
    }
    normalized = {key: tuple(value) if key in tuple_fields else value for key, value in values.items()}
    return cls(**normalized)


def load_config(path: str | Path) -> AppConfig:
    """Read YAML into typed dataclasses and validate cross-field constraints."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError("Root YAML value must be a mapping")
    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    sections = {
        name: _coerce_section(name, raw.get(name, {}), cls)
        for name, cls in _SECTION_TYPES.items()
    }
    config = AppConfig(**sections)
    config.validate()
    return config


def save_config(config: AppConfig, path: str | Path) -> None:
    """Serialize a validated configuration as stable human-readable YAML."""
    config.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False, allow_unicode=True)
