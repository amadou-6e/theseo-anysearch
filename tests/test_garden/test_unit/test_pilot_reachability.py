"""Unit tests for the reachability / false-open probe redesign (F4)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from theseo_anysearch.garden.evaluation.reachability import (
    FalseOpenVeto,
    calibrate_decision_threshold,
    derive_false_open_veto,
    false_open_false_closed,
    per_bin_auprc,
    sample_reachability_pairs,
    two_way_agreement,
)

_SIX = ndimage.generate_binary_structure(3, 1)


def _chokepoint_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One free component joined through a single gap in a wall."""

    occ = np.zeros((1, 5, 11), dtype=np.uint8)
    occ[0, :, 5] = 1
    occ[0, 2, 5] = 0  # the only gap
    valid = np.ones_like(occ, dtype=bool)
    free = valid & ~occ.astype(bool)
    labels, _ = ndimage.label(free, structure=_SIX)
    assert labels.max() == 1
    return occ, valid, labels.astype(np.int32)


def _two_room_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two free components separated by a solid one-voxel wall."""

    occ = np.zeros((1, 5, 11), dtype=np.uint8)
    occ[0, :, 5] = 1
    valid = np.ones_like(occ, dtype=bool)
    free = valid & ~occ.astype(bool)
    labels, count = ndimage.label(free, structure=_SIX)
    assert count == 2
    return occ, valid, labels.astype(np.int32)


def test_stratified_sampler_covers_distance_bins_and_labels_are_exact() -> None:
    occ, valid, labels = _chokepoint_grid()
    plan = sample_reachability_pairs(occ, valid, labels, count=60, seed=0, n_bins=4)

    assert len(plan) > 0
    # a stratified positive must be genuinely same-component; a component or
    # boundary negative must be genuinely different-component in the base graph
    for start, goal, reachable, kind in zip(
        plan.starts, plan.goals, plan.reachable, plan.kind
    ):
        same_component = (
            labels[tuple(start)] > 0 and labels[tuple(start)] == labels[tuple(goal)]
        )
        if kind == "stratified_positive":
            assert same_component and reachable
        elif kind == "component_negative":
            assert not same_component and not reachable
    # positives spread across more than one geodesic bin
    positive_bins = {int(b) for b, r in zip(plan.distance_bin, plan.reachable) if r and b >= 0}
    assert len(positive_bins) >= 2
    # obstacle-perturbation hard negatives exist
    assert "boundary_negative" in set(plan.kind)
    for coordinate, occupied, kind in zip(
        plan.perturbation_coordinates, plan.perturbation_occupied, plan.kind
    ):
        if kind == "boundary_negative":
            assert occupied == 1 and np.all(coordinate >= 0)
        elif kind == "boundary_positive":
            assert occupied == 0 and np.all(coordinate >= 0)
        else:
            assert occupied == -1 and np.all(coordinate == -1)


def test_two_room_grid_yields_margin_separated_component_negatives() -> None:
    occ, valid, labels = _two_room_grid()
    plan = sample_reachability_pairs(
        occ, valid, labels, count=48, seed=1, n_bins=3, component_margin=3.0
    )
    negatives = [
        (s, g)
        for s, g, k in zip(plan.starts, plan.goals, plan.kind)
        if k == "component_negative"
    ]
    assert negatives
    for start, goal in negatives:
        assert np.linalg.norm(start - goal) >= 3.0
        assert labels[tuple(start)] != labels[tuple(goal)]
    # hard positives from joining a one-voxel wall
    assert "boundary_positive" in set(plan.kind)


def test_per_bin_auprc_reports_overall_and_each_bin() -> None:
    rng = np.random.default_rng(0)
    reachable = np.array([True] * 40 + [False] * 40)
    distance_bin = np.array([0] * 20 + [1] * 20 + [-1] * 40)
    scores = np.where(reachable, rng.uniform(0.6, 1.0, 80), rng.uniform(0.0, 0.4, 80))
    report = per_bin_auprc(scores, reachable, distance_bin)
    assert report["overall"] > 0.9
    assert "bin_0" in report and "bin_1" in report


def test_calibrate_threshold_recovers_a_separating_operating_point() -> None:
    reachable = np.array([True] * 50 + [False] * 50)
    scores = np.where(reachable, 0.8, 0.2)
    threshold = calibrate_decision_threshold(scores, reachable)
    fo, fc = false_open_false_closed(scores, reachable, threshold)
    assert fo == 0.0 and fc == 0.0


def test_false_open_false_closed_arithmetic() -> None:
    reachable = np.array([True, True, False, False])
    scores = np.array([0.9, 0.1, 0.9, 0.1])  # one false-closed, one false-open
    fo, fc = false_open_false_closed(scores, reachable, threshold=0.5)
    assert fo == pytest.approx(0.5)
    assert fc == pytest.approx(0.5)


def test_two_way_agreement_requires_both_directions() -> None:
    forward = np.array([0.9, 0.9, 0.2])
    backward = np.array([0.9, 0.1, 0.2])
    agreed = two_way_agreement(forward, backward, threshold=0.5)
    assert agreed.tolist() == [True, False, False]


def test_derive_false_open_veto_sits_below_the_best_baseline() -> None:
    veto = derive_false_open_veto(
        {"fixed_random_projection": 0.148, "pca": 0.19}, margin=0.02
    )
    assert isinstance(veto, FalseOpenVeto)
    assert veto.best_baseline == "fixed_random_projection"
    assert veto.threshold == pytest.approx(0.128)
    assert veto.rejects(0.13) is True
    assert veto.rejects(0.10) is False


def test_derive_false_open_veto_requires_a_baseline() -> None:
    with pytest.raises(ValueError, match="at least one baseline"):
        derive_false_open_veto({})
