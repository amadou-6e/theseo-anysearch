"""Graded-occlusion fixture geometries for the reachability denominator screen (R1).

Each fixture is a small procedural voxel world where the *observer* cannot see
part of the geometry: an ``unknown`` slab hides some wall gaps (so a blocked-vs-
open call needs completion) and some solid wall (so an open-vs-blocked call
needs completion). Connectivity labels come from the *completed* free space;
baseline features only see the *observed* free space.

This is a CPU stand-in for the real v2r1 corpus, sized for the comparison screen
in ``experiments/perception_encoder/reachability_redesign_screen.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_SIX = ndimage.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class OccludedGeometry:
    """One fixture world and its true / observed structure."""

    geometry_id: str
    occupancy: np.ndarray  # (D, H, W) bool, completed ground truth
    valid_mask: np.ndarray  # (D, H, W) bool
    unknown: np.ndarray  # (D, H, W) bool, hidden from the observer
    free_component_labels: np.ndarray  # (D, H, W) int, completed free-space components
    observed_occupancy: np.ndarray  # occupancy the observer sees (unknown -> assumed free)

    @property
    def completed_free(self) -> np.ndarray:
        return self.valid_mask & ~self.occupancy

    @property
    def observed_free(self) -> np.ndarray:
        return self.valid_mask & ~self.observed_occupancy & ~self.unknown


def _label(free: np.ndarray) -> np.ndarray:
    labels, _ = ndimage.label(free, structure=_SIX)
    return labels.astype(np.int32)


def generate_occluded_geometry(seed: int, *, side: int = 15) -> OccludedGeometry:
    rng = np.random.default_rng(seed)
    occ = np.zeros((1, side, side), dtype=bool)
    unknown = np.zeros_like(occ)

    # Two interior walls along the W axis. Wall 1 always has a gap; wall 2 is
    # solid ~55% of the time, which is what creates genuine disconnection.
    wall_cols = sorted(rng.choice(np.arange(3, side - 3), size=2, replace=False))
    wall2_solid = bool(rng.random() < 0.55)
    for index, col in enumerate(wall_cols):
        occ[0, :, col] = True
        if index == 0 or not wall2_solid:
            occ[0, int(rng.integers(1, side - 1)), col] = False

    # Occlude a band of columns, biased to sit over wall 2 so the observer
    # cannot tell whether wall 2 is solid (hidden merge / hidden split).
    if rng.random() < 0.7:
        band_start = int(np.clip(wall_cols[1] - rng.integers(0, 3), 1, side - 3))
    else:
        band_start = int(rng.integers(1, max(2, side - 5)))
    band_width = int(rng.integers(2, 5))
    unknown[0, :, band_start : band_start + band_width] = True
    unknown &= ~occ | (occ & unknown)  # unknown may cover wall cells too
    # The observer treats unknown cells as free (optimistic), which is exactly
    # why a raw-observed baseline cannot tell a hidden solid wall from a gap.
    observed_occ = occ & ~unknown

    valid = np.ones_like(occ)
    labels = _label(valid & ~occ)
    return OccludedGeometry(
        geometry_id=f"rfix-{seed:04d}",
        occupancy=occ,
        valid_mask=valid,
        unknown=unknown,
        free_component_labels=labels,
        observed_occupancy=observed_occ,
    )


def occlusion_corpus(n: int = 48, *, seed: int = 0) -> list[OccludedGeometry]:
    """A deterministic corpus with a spread of wall counts and occlusion bands."""

    return [generate_occluded_geometry(seed + index) for index in range(n)]


def shortest_path(free: np.ndarray, start: tuple[int, int, int]) -> np.ndarray:
    """Six-connected BFS distance field from ``start`` over ``free``."""

    from collections import deque

    dist = np.full(free.shape, -1, dtype=np.int32)
    if not free[start]:
        return dist
    dist[start] = 0
    queue = deque([start])
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nb = (x + dx, y + dy, z + dz)
            if any(c < 0 or c >= free.shape[i] for i, c in enumerate(nb)):
                continue
            if free[nb] and dist[nb] < 0:
                dist[nb] = dist[(x, y, z)] + 1
                queue.append(nb)
    return dist


def occlusion_span_along_path(
    geometry: OccludedGeometry, start: tuple[int, int, int], goal: tuple[int, int, int]
) -> int:
    """Count unknown cells on one completed-graph shortest path, or -1 if unreachable."""

    free = geometry.completed_free
    forward = shortest_path(free, start)
    if forward[goal] < 0:
        return -1
    backward = shortest_path(free, goal)
    total = forward[goal]
    on_path = (forward >= 0) & (backward >= 0) & (forward + backward == total)
    return int(np.count_nonzero(on_path & geometry.unknown))
