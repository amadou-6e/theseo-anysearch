"""Regression cases for review #350: path ambiguity and geometry leakage."""
import numpy as np
import pytest

from theseo_anysearch.garden.evaluation.ceilings import geometry_held_out_posteriors
from theseo_anysearch.garden.pilots.reachability_fixtures import (
    OccludedGeometry,
    occlusion_span_along_path,
)


def test_span_counts_one_path_not_all_shortest_path_cells():
    free = np.ones((1, 3, 3), dtype=bool)
    unknown = free.copy()
    unknown[0, 0, 0] = unknown[0, 2, 2] = False
    geometry = OccludedGeometry("fork", ~free, free, unknown, free.astype(int), ~free)
    assert occlusion_span_along_path(geometry, (0, 0, 0), (0, 2, 2)) == 3


def test_span_uses_lexicographic_tie_break():
    free = np.ones((1, 3, 3), dtype=bool)
    unknown = np.zeros_like(free)
    unknown[0, 0, 1] = True
    geometry = OccludedGeometry("fork", ~free, free, unknown, free.astype(int), ~free)
    assert occlusion_span_along_path(geometry, (0, 0, 0), (0, 2, 2)) == 1
    assert occlusion_span_along_path(geometry, (0, 0, 0), (0, 0, 0)) == 0


def test_reference_cannot_use_neighbours_from_held_out_geometry():
    features = np.array([[0.0], [0.0], [100.0], [100.0]])
    labels = np.array([0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b"])
    np.testing.assert_array_equal(
        geometry_held_out_posteriors(features, labels, groups, k=2), [1, 1, 0, 0]
    )


def test_reference_rejects_insufficient_geometry_support():
    with pytest.raises(ValueError, match="two geometry"):
        geometry_held_out_posteriors(np.zeros((2, 1)), np.array([0, 1]), np.array(["a", "a"]), k=1)
    with pytest.raises(ValueError, match="k training rows"):
        geometry_held_out_posteriors(np.zeros((2, 1)), np.array([0, 1]), np.array(["a", "b"]), k=2)
