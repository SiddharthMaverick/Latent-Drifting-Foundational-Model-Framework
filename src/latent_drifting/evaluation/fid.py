"""Frechet distance and streaming activation statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class ActivationStats:
    count: int
    mean: np.ndarray
    covariance: np.ndarray


class StreamingStats:
    """Numerically stable batch-merging mean/covariance accumulator."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, features: torch.Tensor | np.ndarray) -> None:
        values = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("Activation batches must have shape [samples, features]")
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        assert self.mean is not None and self.m2 is not None
        delta = batch_mean - self.mean
        combined = self.count + batch_count
        self.m2 = self.m2 + batch_m2 + np.outer(delta, delta) * self.count * batch_count / combined
        self.mean = self.mean + delta * batch_count / combined
        self.count = combined

    def finalize(self) -> ActivationStats:
        if self.count < 2 or self.mean is None or self.m2 is None:
            raise RuntimeError("At least two activations are required for covariance")
        return ActivationStats(self.count, self.mean.copy(), self.m2 / (self.count - 1))


def _symmetric_matrix_sqrt(matrix: np.ndarray, eps: float = 0.0) -> np.ndarray:
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, eps, None)
    return (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T


def frechet_distance(
    mean_real: np.ndarray,
    covariance_real: np.ndarray,
    mean_generated: np.ndarray,
    covariance_generated: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate the standard Gaussian Frechet/Inception distance."""
    mean_real = np.asarray(mean_real, dtype=np.float64)
    mean_generated = np.asarray(mean_generated, dtype=np.float64)
    cov_real = np.asarray(covariance_real, dtype=np.float64)
    cov_generated = np.asarray(covariance_generated, dtype=np.float64)
    if mean_real.shape != mean_generated.shape or cov_real.shape != cov_generated.shape:
        raise ValueError("Real and generated statistics must have matching dimensions")
    root_real = _symmetric_matrix_sqrt(cov_real)
    middle = root_real @ cov_generated @ root_real
    middle_root = _symmetric_matrix_sqrt(middle)
    difference = mean_real - mean_generated
    value = difference @ difference + np.trace(cov_real) + np.trace(cov_generated) - 2 * np.trace(middle_root)
    return float(max(value, 0.0))


def save_stats(stats: ActivationStats, path: str) -> None:
    np.savez_compressed(path, count=stats.count, mean=stats.mean, covariance=stats.covariance)


def load_stats(path: str) -> ActivationStats:
    with np.load(path) as values:
        return ActivationStats(int(values["count"]), values["mean"], values["covariance"])
