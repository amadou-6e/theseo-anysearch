"""Regression cases for review #350: path ambiguity and geometry leakage."""
import numpy as np
import pytest
import json
from pathlib import Path

from theseo_anysearch.garden.evaluation.ceilings import geometry_held_out_posteriors
from theseo_anysearch.garden.pilots.reachability_fixtures import (
    OccludedGeometry,
    occlusion_span_along_path,
)
from theseo_anysearch.garden.pilots.io import payload_sha256


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


@pytest.mark.parametrize("k", [1, np.int32(1), np.int64(1), np.uint64(1)])
def test_reference_accepts_integer_scalars(k):
    np.testing.assert_array_equal(
        geometry_held_out_posteriors(np.array([[0.0], [1.0]]),
                                    np.array([0, 1]), np.array(["a", "b"]), k=k),
        [1.0, 0.0],
    )


@pytest.mark.parametrize("k", [True, np.bool_(True), 1.0, np.float64(1.0), 0, -1])
def test_reference_rejects_nonpositive_or_noninteger_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        geometry_held_out_posteriors(np.zeros((2, 1)), np.array([0, 1]),
                                    np.array(["a", "b"]), k=k)


def test_span_floods_once_and_handles_blocked_endpoints(monkeypatch):
    from theseo_anysearch.garden.pilots import reachability_fixtures as fixtures

    calls = []
    original = fixtures.shortest_path

    def counted(free, start):
        calls.append(start)
        return original(free, start)

    monkeypatch.setattr(fixtures, "shortest_path", counted)
    free = np.ones((1, 3, 3), dtype=bool)
    free[0, 0, 0] = False
    g = OccludedGeometry("blocked", ~free, np.ones_like(free),
                         np.zeros_like(free), free.astype(int), ~free)
    assert occlusion_span_along_path(g, (0, 0, 0), (0, 2, 2)) == -1
    assert calls == [(0, 2, 2)]
    calls.clear()
    assert occlusion_span_along_path(g, (0, 2, 2), (0, 0, 0)) == -1
    assert calls == [(0, 0, 0)]


@pytest.mark.parametrize("relative", [
    "v2r1_p0c/p0c-report.json", "v2r1_p0d/p0d-report.json",
    "v2r1_p1/p1-terminal-report.json",
])
def test_shared_hash_preserves_archived_report_identities(relative):
    root = Path(__file__).resolve().parents[3]
    report = json.loads((root / "experiments/perception_encoder/results" / relative).read_text())
    expected = report.pop("report_payload_sha256")
    assert payload_sha256(report) == expected


def test_hash_is_order_independent_and_rejects_nonfinite_values():
    assert payload_sha256({"b": 1, "a": "\u00e9"}) == payload_sha256({"a": "\u00e9", "b": 1})
    with pytest.raises(ValueError):
        payload_sha256({"value": float("nan")})
