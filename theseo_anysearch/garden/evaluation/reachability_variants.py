"""Feature and sample builders for the reachability denominator screen (R1-R3).

Given an :class:`OccludedGeometry`, produce the pair sets and feature matrices
the comparison screen needs:

- ``sample_occlusion_stratified_pairs`` - reachable label from the *completed*
  free-space graph, plus the occlusion span along the path;
- ``raw_observed_feature`` - local occupancy the observer sees (unknown as a
  channel); the baseline / floor input;
- ``rich_completed_feature`` - local *completed* occupancy plus geodesic scalars;
  a linearly-strong stand-in for a good encoder / the Bayes-ceiling input;
- ``coordinates_null`` - normalized coordinates only;
- ``pair_matrix`` - symmetric start/goal combination like the real pair probe;
- ``component_label_map`` / ``observed_component_label_map`` - per-cell free-space
  component labels for the structure metric (option C).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theseo_anysearch.garden.pilots.reachability_fixtures import (
    OccludedGeometry,
    occlusion_span_along_path,
    shortest_path,
)

_PATCH = 2  # +/- radius of the local feature patch


@dataclass(frozen=True)
class PairSample:
    starts: np.ndarray  # (N, 3) int
    goals: np.ndarray  # (N, 3) int
    reachable: np.ndarray  # (N,) bool, completed-graph connectivity
    occlusion_span: np.ndarray  # (N,) int, unknown cells on the path (-1 if unreachable)
    geometry_ids: tuple[str, ...]


def _free_cells(geometry: OccludedGeometry) -> np.ndarray:
    return np.argwhere(geometry.completed_free)


def sample_occlusion_stratified_pairs(
    geometry: OccludedGeometry, *, count: int, seed: int
) -> PairSample:
    rng = np.random.default_rng(seed)
    cells = _free_cells(geometry)
    if len(cells) < 4:
        raise ValueError("fixture has too few free cells")
    labels = geometry.free_component_labels
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    reachable: list[bool] = []
    span: list[int] = []
    # aim for a mix of reachable (varied occlusion span) and unreachable pairs
    for _ in range(count * 6):
        if len(starts) >= count:
            break
        a = cells[int(rng.integers(0, len(cells)))]
        b = cells[int(rng.integers(0, len(cells)))]
        if np.array_equal(a, b):
            continue
        is_reachable = bool(
            labels[tuple(a)] > 0 and labels[tuple(a)] == labels[tuple(b)]
        )
        s = occlusion_span_along_path(geometry, tuple(int(v) for v in a), tuple(int(v) for v in b))
        starts.append(a)
        goals.append(b)
        reachable.append(is_reachable)
        span.append(s)
    if len(starts) < count // 2:
        raise ValueError("could not sample enough pairs")
    return PairSample(
        starts=np.stack(starts).astype(np.int64),
        goals=np.stack(goals).astype(np.int64),
        reachable=np.asarray(reachable, dtype=bool),
        occlusion_span=np.asarray(span, dtype=np.int64),
        geometry_ids=(geometry.geometry_id,) * len(starts),
    )


def _patch(volume: np.ndarray, coord: np.ndarray) -> np.ndarray:
    padded = np.pad(volume.astype(np.float64), _PATCH, mode="edge")
    x, y, z = (int(v) + _PATCH for v in coord)
    window = padded[
        x - _PATCH : x + _PATCH + 1,
        y - _PATCH : y + _PATCH + 1,
        z - _PATCH : z + _PATCH + 1,
    ]
    return window.reshape(-1)


def raw_observed_feature(geometry: OccludedGeometry, coord: np.ndarray) -> np.ndarray:
    """What a raw-observed baseline sees: observed occupancy + an unknown channel."""

    return np.concatenate(
        [_patch(geometry.observed_occupancy, coord), _patch(geometry.unknown, coord)]
    )


def rich_completed_feature(geometry: OccludedGeometry, coord: np.ndarray) -> np.ndarray:
    """Completion-aware: the true completed occupancy plus a geodesic scalar.

    Stands in for an encoder that learned occupancy completion / connectivity.
    """

    field = shortest_path(geometry.completed_free, tuple(int(v) for v in coord))
    reach = (field >= 0).astype(np.float64)
    return np.concatenate(
        [
            _patch(geometry.occupancy, coord),
            _patch(reach, coord),
            [float(geometry.free_component_labels[tuple(coord)])],
        ]
    )


def coordinates_null(geometry: OccludedGeometry, coord: np.ndarray) -> np.ndarray:
    side = np.asarray(geometry.occupancy.shape, dtype=np.float64)
    return (np.asarray(coord, dtype=np.float64) * 2.0 / np.maximum(side - 1, 1)) - 1.0


def pair_matrix(
    feature_fn, geometry: OccludedGeometry, sample: PairSample
) -> np.ndarray:
    rows = []
    for start, goal in zip(sample.starts, sample.goals):
        fs = feature_fn(geometry, start)
        fg = feature_fn(geometry, goal)
        rows.append(np.concatenate([fs + fg, np.abs(fs - fg), fs * fg]))
    return np.stack(rows)


def component_label_map(geometry: OccludedGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell true completed component label over observed-free cells, and the mask."""

    mask = geometry.observed_free
    return geometry.free_component_labels[mask], mask


def observed_component_features(
    geometry: OccludedGeometry, *, which: str
) -> np.ndarray:
    """Per-observed-free-cell feature matrix for the structure metric (option C)."""

    coords = np.argwhere(geometry.observed_free)
    fn = raw_observed_feature if which == "raw" else rich_completed_feature
    return np.stack([fn(geometry, coord) for coord in coords])
