"""Analytic fixtures for pilot metrics, statistics, and learning curves."""
from __future__ import annotations

import numpy as np
import pytest

from theseo_anysearch.garden.evaluation.curves import (
    backtest_learning_curves,
    extrapolation_can_rescue,
    predict_learning_curve,
    target_horizon,
)
from theseo_anysearch.garden.evaluation.metrics import (
    adapted_rand_error,
    binary_ranking_metrics,
    binary_iou,
    boundary_f1,
    collapse_diagnostics,
    connectivity_change_fraction,
    cubical_betti_numbers,
    linear_discriminant_rank,
    macro_f1,
    normalized_mae,
    raw_and_l2_collapse_diagnostics,
    topology_reconstruction_metrics,
    variation_of_information,
)
from theseo_anysearch.garden.evaluation.statistics import (
    interquartile_mean,
    paired_stratified_geometry_bootstrap,
    performance_profile,
)


def test_classification_and_boundary_metrics_have_known_extremes() -> None:
    target = np.asarray([0, 0, 1, 1])
    assert binary_iou(target == 1, target == 1) == 1
    assert binary_iou(target == 1, target == 0) == 0
    assert macro_f1(target, target, classes=(0, 1)) == 1
    boundary = np.zeros((5, 5, 5), dtype=bool)
    boundary[2, 2, 2] = True
    shifted = np.roll(boundary, 1, axis=0)
    assert boundary_f1(shifted, boundary, tolerance=1) == 1
    assert boundary_f1(shifted, boundary, tolerance=0) == 0


def test_partition_metrics_are_zero_only_for_matching_partitions() -> None:
    target = np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]])
    same_partition = np.asarray([[8, 8, 4, 4], [8, 8, 4, 4]])
    merged = np.ones_like(target)
    assert variation_of_information(same_partition, target) == pytest.approx(0)
    assert adapted_rand_error(same_partition, target) == pytest.approx(0)
    assert variation_of_information(merged, target) > 0
    assert adapted_rand_error(merged, target) > 0


def test_ranking_and_error_metrics_expose_false_open_predictions() -> None:
    target = np.asarray([0, 0, 1, 1], dtype=bool)
    perfect = binary_ranking_metrics(np.asarray([0.1, 0.2, 0.8, 0.9]), target)
    assert perfect.auroc == perfect.auprc == perfect.balanced_accuracy == 1
    assert perfect.false_open_rate == perfect.false_closed_rate == 0
    broken_wall = binary_ranking_metrics(np.asarray([0.9, 0.2, 0.8, 0.9]), target)
    assert broken_wall.false_open_rate == 0.5
    assert normalized_mae([1, 3], [2, 2], normalizer=2) == 0.5


def test_cubical_betti_numbers_detect_components_loops_and_cavities() -> None:
    single = np.zeros((5, 5, 5), dtype=bool)
    single[2, 2, 2] = True
    assert cubical_betti_numbers(single) == (1, 0, 0)

    disconnected = single.copy()
    disconnected[0, 0, 0] = True
    assert cubical_betti_numbers(disconnected) == (2, 0, 0)

    ring = np.zeros((5, 5, 5), dtype=bool)
    ring[1:4, 1, 2] = True
    ring[1:4, 3, 2] = True
    ring[1, 1:4, 2] = True
    ring[3, 1:4, 2] = True
    assert cubical_betti_numbers(ring) == (1, 1, 0)

    shell = np.ones((3, 3, 3), dtype=bool)
    shell[1, 1, 1] = False
    assert cubical_betti_numbers(shell) == (1, 0, 1)


def test_topology_reconstruction_bundle_detects_a_broken_wall() -> None:
    target = np.zeros((5, 5, 5), dtype=bool)
    target[2, :, :] = True
    prediction = target.copy()
    prediction[2, 2, 2] = False
    valid = np.ones_like(target)
    left = np.ravel_multi_index((1, 2, 2), target.shape)
    right = np.ravel_multi_index((3, 2, 2), target.shape)
    pairs = np.asarray([[left, right]])
    metrics = topology_reconstruction_metrics(
        prediction, target, valid_mask=valid, pairs=pairs
    )
    assert metrics.component_count_error == 1
    assert metrics.connectivity_change_fraction == 1
    assert metrics.normalized_variation_of_information > 0
    assert connectivity_change_fraction(
        np.asarray([1, 1, 1]), np.asarray([1, 1, 2]), np.asarray([[0, 2]])
    ) == 1


def test_collapse_diagnostics_separate_full_rank_and_collapsed_features() -> None:
    rng = np.random.default_rng(8)
    full = rng.normal(size=(256, 16))
    collapsed = np.repeat(rng.normal(size=(256, 1)), 16, axis=1)
    full_metrics = collapse_diagnostics(full)
    collapsed_metrics = collapse_diagnostics(collapsed)
    assert full_metrics.effective_rank > 12
    assert collapsed_metrics.effective_rank == pytest.approx(1)
    assert collapsed_metrics.dominant_component_share == pytest.approx(1)
    paired = raw_and_l2_collapse_diagnostics(full)
    assert set(paired) == {"raw", "l2_normalized"}
    assert len(paired["raw"].singular_values) == 16

    views = np.repeat([0, 1, 2], 20)
    view_features = rng.normal(scale=0.05, size=(60, 4))
    view_features[:, 0] += views
    assert linear_discriminant_rank(view_features, views) >= 1


def test_paired_bootstrap_recovers_difference_and_identical_zero() -> None:
    rows = 24
    geometry_ids = [f"g-{index}" for index in range(rows)]
    families = [f"family-{index % 4}" for index in range(rows)]
    bands = [f"band-{index % 3}" for index in range(rows)]
    reference = np.linspace(0.2, 0.6, rows)
    candidate = reference + 0.1
    result = paired_stratified_geometry_bootstrap(
        candidate,
        reference,
        geometry_ids=geometry_ids,
        geometry_families=families,
        occupancy_bands=bands,
        resamples=1_000,
        seed=12,
    )
    assert result.mean_difference == pytest.approx(0.1)
    assert result.lower_95 > 0
    assert result.probability_of_improvement == 1
    identical = paired_stratified_geometry_bootstrap(
        reference,
        reference,
        geometry_ids=geometry_ids,
        geometry_families=families,
        occupancy_bands=bands,
        resamples=100,
    )
    assert identical.lower_95 == identical.upper_95 == 0


def test_iqm_and_performance_profile_have_analytic_values() -> None:
    assert interquartile_mean([0, 1, 2, 100]) == pytest.approx(1.5)
    np.testing.assert_allclose(
        performance_profile([0.2, 0.5, 0.9], [0.0, 0.5, 1.0]), [1.0, 2 / 3, 0.0]
    )


def test_learning_curve_prediction_horizon_and_calibration_fixture() -> None:
    updates = [100, 200, 400, 800, 1600]
    scores = [0.8 - 2 / np.sqrt(update) for update in updates]
    prediction = predict_learning_curve(
        updates[:3], scores[:3], target_update=updates[-1], seed=4
    )
    assert prediction.mean == pytest.approx(scores[-1], abs=0.05)
    assert prediction.lower_95 < scores[-1] < prediction.upper_95
    calibration = backtest_learning_curves([(updates, scores)] * 10, seed=4)
    assert calibration.hidden_points == 20
    assert calibration.calibrated
    assert target_horizon(2_000, 10_000) == 8_000
    assert extrapolation_can_rescue(prediction, prediction)
