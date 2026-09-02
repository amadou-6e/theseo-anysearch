"""Geometry-clustered statistical comparisons for perception pilots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_difference: float
    lower_95: float
    upper_95: float
    probability_of_improvement: float
    resamples: int
    seed: int
    sample_indices: np.ndarray


def paired_stratified_geometry_bootstrap(
    candidate_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    geometry_ids: Sequence[str],
    geometry_families: Sequence[str],
    occupancy_bands: Sequence[str],
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Resample complete paired geometries within joint family/density strata."""

    candidate = np.asarray(candidate_scores, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    rows = len(candidate)
    metadata = (reference, geometry_ids, geometry_families, occupancy_bands)
    if rows == 0 or any(len(values) != rows for values in metadata):
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    if len(set(geometry_ids)) != rows:
        raise ValueError("bootstrap rows must be unique complete geometry IDs")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("bootstrap scores must be finite")

    strata: dict[tuple[str, str], np.ndarray] = {}
    for index, key in enumerate(zip(geometry_families, occupancy_bands)):
        strata.setdefault(key, []).append(index)  # type: ignore[arg-type]
    strata = {key: np.asarray(indices, dtype=np.int64) for key, indices in strata.items()}
    rng = np.random.default_rng(seed)
    sampled = np.empty((resamples, rows), dtype=np.int32)
    column = 0
    for key in sorted(strata):
        indices = strata[key]
        width = len(indices)
        sampled[:, column : column + width] = rng.choice(
            indices, size=(resamples, width), replace=True
        )
        column += width
    differences = candidate - reference
    bootstrap_means = differences[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return PairedBootstrapResult(
        mean_difference=float(differences.mean()),
        lower_95=float(lower),
        upper_95=float(upper),
        probability_of_improvement=float(np.mean(bootstrap_means > 0)),
        resamples=resamples,
        seed=seed,
        sample_indices=sampled,
    )


def interquartile_mean(values: Sequence[float]) -> float:
    """Mean of the central 50%, with fractional boundary weighting."""

    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if len(sorted_values) == 0:
        raise ValueError("IQM requires at least one value")
    positions = (np.arange(len(sorted_values)) + 0.5) / len(sorted_values)
    weights = np.minimum(positions + 0.5 / len(sorted_values), 0.75) - np.maximum(
        positions - 0.5 / len(sorted_values), 0.25
    )
    weights = np.maximum(weights, 0)
    return float(np.average(sorted_values, weights=weights))


def performance_profile(values: Sequence[float], thresholds: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("performance profiles require finite scores")
    return np.asarray([np.mean(scores >= threshold) for threshold in thresholds])
