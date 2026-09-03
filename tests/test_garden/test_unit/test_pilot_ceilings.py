"""Unit tests for model-free denominator ceilings (F1)."""
from __future__ import annotations

import numpy as np
import pytest

from theseo_anysearch.garden.evaluation.ceilings import (
    NON_COLLAPSE_MIN_FRACTION,
    bayes_error_direct,
    bayes_error_knn,
    bayes_error_mst,
    ceiling_effective_rank_fraction,
    classification_metric_ceiling,
    knn_loo_posteriors,
    metric_ceiling_method,
    regression_metric_ceiling,
)
from theseo_anysearch.garden.evaluation.ceilings import _loo_neighbour_indices


def _separable(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    positive = rng.normal(loc=(6.0, 0.0), scale=0.4, size=(half, 2))
    negative = rng.normal(loc=(-6.0, 0.0), scale=0.4, size=(half, 2))
    features = np.vstack([positive, negative])
    labels = np.array([1] * half + [0] * half)
    order = rng.permutation(n)
    return features[order], labels[order]


def _pure_noise(n: int = 400, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 4))
    labels = rng.integers(0, 2, size=n)
    return features, labels


def test_loo_neighbours_never_return_self() -> None:
    features, _ = _separable(120, seed=3)
    neighbours = _loo_neighbour_indices(features, k=10)
    assert neighbours.shape == (120, 10)
    assert not any(row_index in neighbours[row_index] for row_index in range(120))


def test_bayes_error_estimators_agree_near_zero_on_separable_data() -> None:
    features, labels = _separable()
    assert bayes_error_knn(features, labels) < 0.02
    assert bayes_error_direct(features, labels) < 0.02
    assert bayes_error_mst(features, labels) < 0.05


def test_bayes_error_estimators_are_near_half_on_pure_noise() -> None:
    features, labels = _pure_noise()
    assert 0.35 <= bayes_error_knn(features, labels) <= 0.65
    assert 0.30 <= bayes_error_direct(features, labels) <= 0.55


def test_classification_ceiling_recovers_a_high_value_when_separable() -> None:
    features, labels = _separable()
    iou = classification_metric_ceiling(features, labels, metric="occupied_iou")
    auprc = classification_metric_ceiling(features, labels, metric="reachability_auprc")
    assert iou.value > 0.95
    assert auprc.value > 0.95
    assert iou.method == "bayes_error_knn"
    assert iou.non_collapse is True
    assert iou.sample_count == features.shape[0]


def test_classification_ceiling_is_modest_when_the_task_carries_no_signal() -> None:
    features, labels = _pure_noise()
    auprc = classification_metric_ceiling(features, labels, metric="reachability_auprc")
    # AUPRC of an uninformative ranker sits near the positive base rate (~0.5).
    assert 0.35 <= auprc.value <= 0.7


def test_regression_ceiling_is_tight_for_a_smooth_target() -> None:
    rng = np.random.default_rng(4)
    features = rng.uniform(-1.0, 1.0, size=(400, 3))
    targets = np.sin(features[:, 0]) + 0.05 * rng.normal(size=400)
    ceiling = regression_metric_ceiling(features, targets, normalizer=1.0)
    assert ceiling.method == "knn_residual"
    assert ceiling.value < 0.15


def test_regression_ceiling_reflects_irreducible_noise() -> None:
    rng = np.random.default_rng(5)
    features = rng.uniform(-1.0, 1.0, size=(400, 3))
    targets = rng.normal(scale=1.0, size=400)
    ceiling = regression_metric_ceiling(features, targets, normalizer=1.0)
    # A kNN mean over unrelated neighbours cannot beat predicting the global mean.
    assert ceiling.value > 0.5


def test_effective_rank_fraction_flags_a_collapsed_reference() -> None:
    rng = np.random.default_rng(6)
    basis = rng.normal(size=(2, 64))
    collapsed = rng.normal(size=(300, 2)) @ basis + 1e-4 * rng.normal(size=(300, 64))
    fraction, non_collapse = ceiling_effective_rank_fraction(collapsed)
    assert fraction < NON_COLLAPSE_MIN_FRACTION
    assert non_collapse is False


def test_effective_rank_fraction_accepts_an_isotropic_reference() -> None:
    rng = np.random.default_rng(7)
    isotropic = rng.normal(size=(300, 32))
    fraction, non_collapse = ceiling_effective_rank_fraction(isotropic)
    assert fraction > NON_COLLAPSE_MIN_FRACTION
    assert non_collapse is True


def test_knn_posteriors_require_both_classes() -> None:
    features = np.random.default_rng(8).normal(size=(60, 3))
    with pytest.raises(ValueError, match="both classes"):
        knn_loo_posteriors(features, np.ones(60, dtype=int))


def test_metric_ceiling_method_routing() -> None:
    assert metric_ceiling_method("occupied_iou") == "bayes_error_knn"
    assert metric_ceiling_method("geodesic_nmae") == "knn_residual"
    with pytest.raises(ValueError):
        metric_ceiling_method("not_a_component")
