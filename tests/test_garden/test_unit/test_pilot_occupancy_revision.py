"""Tests for the held-out occupancy redesign (F3)."""
from __future__ import annotations

import numpy as np

from theseo_anysearch.garden.evaluation.occupancy import (
    heldout_occupancy_queries,
    local_context_features,
)


def _volume() -> tuple[np.ndarray, np.ndarray]:
    occupancy = np.zeros((9, 9, 9), dtype=bool)
    occupancy[2:7, 4, 2:7] = True
    occupancy[4, 2:7, 2:7] = True
    unknown = np.zeros_like(occupancy)
    unknown[[0, -1], :, :] = True
    return occupancy, unknown


def test_heldout_query_cells_are_absent_from_encoder_input() -> None:
    occupancy, unknown = _volume()
    plan = heldout_occupancy_queries(occupancy, unknown, count=40, seed=3)
    unique = np.unique(plan.coordinates, axis=0)
    assert not plan.input_occupancy[tuple(unique.T)].any()
    assert plan.input_unknown[tuple(unique.T)].all()
    assert set(plan.targets) == {0.0, 1.0}
    assert not np.shares_memory(plan.input_occupancy, occupancy)


def test_off_grid_queries_are_deterministic_and_subvoxel() -> None:
    occupancy, unknown = _volume()
    first = heldout_occupancy_queries(
        occupancy, unknown, count=20, seed=7, off_grid=True
    )
    second = heldout_occupancy_queries(
        occupancy, unknown, count=20, seed=7, off_grid=True
    )
    assert np.array_equal(first.normalized_coordinates, second.normalized_coordinates)
    grid_positions = (first.normalized_coordinates + 1) * 4
    assert np.any(np.abs(grid_positions - np.round(grid_positions)) > 1e-4)


def test_cross_channel_removes_the_occupied_identity_path() -> None:
    occupancy, unknown = _volume()
    plan = heldout_occupancy_queries(
        occupancy, unknown, count=20, seed=1, cross_channel=True
    )
    assert not plan.input_occupancy.any()
    assert plan.input_unknown[occupancy].all()


def test_context_features_zero_the_query_center() -> None:
    occupancy, unknown = _volume()
    plan = heldout_occupancy_queries(occupancy, unknown, count=20, seed=9)
    features = local_context_features(
        plan.input_occupancy, plan.input_unknown, plan.coordinates, radius=1
    )
    centers = features.reshape(20, 3, 3, 3, 3)[:, :, 1, 1, 1]
    assert np.count_nonzero(centers) == 0
    assert np.isfinite(features).all()
