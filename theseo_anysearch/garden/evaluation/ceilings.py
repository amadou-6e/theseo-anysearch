"""Model-free denominator ceilings for the amended P0C calibration (F1).

The v1/v2 anchor used a trained supervised classifier's penultimate features
as the "encoder ceiling". Neural collapse guarantees those features drop to
roughly (#classes - 1) dimensions, so the ceiling was untrustworthy and P0C
produced ceilings below the baseline floor.

This module estimates the achievable ceiling directly from data with
nonparametric estimators, so there is no reference network to collapse or
diverge:

- ``bayes_error_knn`` / ``bayes_error_direct`` / ``bayes_error_mst`` estimate
  the irreducible error of a binary task;
- ``classification_metric_ceiling`` turns a leave-one-out kNN Bayes proxy into
  an achievable value for a target metric (occupied IoU, reachability AUPRC);
- ``regression_metric_ceiling`` gives an achievable normalized-MAE ceiling for
  clearance / geodesic regression;
- ``ceiling_effective_rank_fraction`` is the non-collapse guard for any
  ceiling that is still derived from a trained reference (F6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree

from theseo_anysearch.garden.evaluation.metrics import (
    binary_iou,
    binary_ranking_metrics,
    collapse_diagnostics,
    normalized_mae,
)

NON_COLLAPSE_MIN_FRACTION = 0.30
DEFAULT_K = 15

MODEL_FREE_METHODS = ("bayes_error_knn", "bayes_error_direct", "bayes_error_mst", "knn_residual")
_CLASSIFICATION_METRICS = ("occupied_iou", "boundary_f1", "reachability_auprc")
_REGRESSION_METRICS = ("clearance_nmae", "geodesic_nmae")


@dataclass(frozen=True)
class CeilingEstimate:
    """One achievable ceiling and the evidence behind it."""

    value: float
    method: str
    k: int
    sample_count: int
    effective_rank_fraction: float | None = None
    non_collapse: bool | None = None


def _features_and_labels(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be (N, D) and aligned with N labels")
    if x.shape[0] < DEFAULT_K + 1:
        raise ValueError(f"model-free ceilings need at least {DEFAULT_K + 1} samples")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    scale = x.std(axis=0, keepdims=True)
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(scale, 1e-9)
    return x, y


def _loo_neighbour_indices(features: np.ndarray, k: int) -> np.ndarray:
    """Return the k nearest *other* rows for every row."""

    if k < 1 or k >= features.shape[0]:
        raise ValueError("k must be in [1, N)")
    tree = cKDTree(features)
    _, indices = tree.query(features, k=k + 1)
    indices = np.atleast_2d(indices)
    # Drop the self match (first column) unless a duplicate point stole it.
    trimmed = np.empty((features.shape[0], k), dtype=np.int64)
    for row in range(features.shape[0]):
        neighbours = [idx for idx in indices[row] if idx != row][:k]
        if len(neighbours) < k:
            neighbours += [idx for idx in indices[row] if idx == row][: k - len(neighbours)]
        trimmed[row] = neighbours
    return trimmed


def knn_loo_posteriors(
    features: np.ndarray, labels: np.ndarray, *, k: int = DEFAULT_K
) -> np.ndarray:
    """Leave-one-out kNN posterior P(y=1 | x) for a binary task."""

    x, y = _features_and_labels(features, labels)
    positive = np.asarray(y, dtype=bool)
    if positive.all() or (~positive).all():
        raise ValueError("kNN posteriors require both classes present")
    neighbours = _loo_neighbour_indices(x, k)
    return positive[neighbours].mean(axis=1)


def bayes_error_knn(
    features: np.ndarray, labels: np.ndarray, *, k: int = DEFAULT_K
) -> float:
    """Leave-one-out kNN majority-vote error rate."""

    posteriors = knn_loo_posteriors(features, labels, k=k)
    predicted = posteriors >= 0.5
    return float(np.mean(predicted != np.asarray(labels, dtype=bool)))


def bayes_error_direct(
    features: np.ndarray, labels: np.ndarray, *, k: int = DEFAULT_K
) -> float:
    """Direct Bayes-error estimate: mean of min(eta, 1 - eta) over kNN posteriors."""

    posteriors = knn_loo_posteriors(features, labels, k=k)
    return float(np.mean(np.minimum(posteriors, 1.0 - posteriors)))


def bayes_error_mst(features: np.ndarray, labels: np.ndarray) -> float:
    """Friedman-Rafsky / Henze-Penrose Bayes-error estimate from a Euclidean MST.

    ``R`` is the number of MST edges joining opposite classes; ``R / n`` is a
    consistent estimate of an upper bound on the Bayes error.
    """

    x, y = _features_and_labels(features, labels)
    positive = np.asarray(y, dtype=bool)
    if positive.all() or (~positive).all():
        raise ValueError("MST Bayes error requires both classes present")
    differences = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))
    tree = minimum_spanning_tree(distances).tocoo()
    cross_class = np.count_nonzero(positive[tree.row] != positive[tree.col])
    return float(cross_class / x.shape[0])


def classification_metric_ceiling(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    k: int = DEFAULT_K,
    method: str = "bayes_error_knn",
) -> CeilingEstimate:
    """Achievable value of a classification-derived metric via a kNN Bayes proxy."""

    if metric not in _CLASSIFICATION_METRICS:
        raise ValueError(f"unknown classification metric {metric!r}")
    if method not in ("bayes_error_knn", "bayes_error_direct"):
        raise ValueError("classification ceilings use a kNN Bayes proxy")
    posteriors = knn_loo_posteriors(features, labels, k=k)
    target = np.asarray(labels, dtype=bool)
    if metric == "occupied_iou":
        value = binary_iou(posteriors >= 0.5, target)
    elif metric == "boundary_f1":
        prediction = posteriors >= 0.5
        true_positive = np.count_nonzero(prediction & target)
        denominator = np.count_nonzero(prediction) + np.count_nonzero(target)
        value = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
    else:
        value = binary_ranking_metrics(posteriors, target).auprc
    return CeilingEstimate(
        value=float(value),
        method=method,
        k=k,
        sample_count=int(features.shape[0]),
        non_collapse=True,
    )


def regression_metric_ceiling(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    normalizer: float,
    k: int = DEFAULT_K,
) -> CeilingEstimate:
    """Achievable normalized-MAE ceiling via leave-one-out kNN regression."""

    x, y = _features_and_labels(features, targets)
    y = np.asarray(y, dtype=np.float64)
    neighbours = _loo_neighbour_indices(x, k)
    predicted = y[neighbours].mean(axis=1)
    value = normalized_mae(predicted, y, normalizer=normalizer)
    return CeilingEstimate(
        value=float(value),
        method="knn_residual",
        k=k,
        sample_count=int(x.shape[0]),
        non_collapse=True,
    )


def ceiling_effective_rank_fraction(embeddings: np.ndarray) -> tuple[float, bool]:
    """Non-collapse guard for a reference-derived ceiling (F6).

    Returns the effective-rank fraction and whether it clears
    ``NON_COLLAPSE_MIN_FRACTION``. The P0C supervised reference measured 0.011.
    """

    fraction = float(collapse_diagnostics(np.asarray(embeddings)).effective_rank_fraction)
    return fraction, fraction > NON_COLLAPSE_MIN_FRACTION


def metric_ceiling_method(metric: str) -> str:
    """Default model-free ceiling method for a pilot-score component."""

    if metric in _CLASSIFICATION_METRICS:
        return "bayes_error_knn"
    if metric in _REGRESSION_METRICS:
        return "knn_residual"
    raise ValueError(f"unknown pilot-score component {metric!r}")


def requires_trained_reference(methods: list[str] | tuple[str, ...]) -> bool:
    """Return whether any active ceiling method requires a trained reference."""

    known = set(MODEL_FREE_METHODS) | {"multitask_reference", "regularized_reference"}
    unknown = set(methods) - known
    if unknown:
        raise ValueError(f"unknown ceiling methods: {sorted(unknown)}")
    return any(method not in MODEL_FREE_METHODS for method in methods)
