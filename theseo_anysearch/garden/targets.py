"""Exact voxel-geometry targets used by pilot probes and pretraining heads."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


SIX_CONNECTED = ndimage.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class GeometryTargets:
    """Deterministic dense and topology targets for one voxel observation."""

    occupancy: np.ndarray
    valid_mask: np.ndarray
    boundary: np.ndarray
    signed_distance: np.ndarray
    free_component_labels: np.ndarray
    free_degree: np.ndarray
    dead_ends: np.ndarray
    junctions: np.ndarray
    free_components: int
    graph_cycle_rank: int


def _validate_volume(volume: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(volume, dtype=bool)
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise ValueError(f"{name} must be one cubic 3D volume")
    return array


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    return ndimage.convolve(mask.astype(np.int16), SIX_CONNECTED.astype(np.int16), mode="constant")


def compute_geometry_targets(
    occupancy: np.ndarray,
    *,
    unknown_mask: np.ndarray | None = None,
    truncation: float = 8.0,
) -> GeometryTargets:
    """Compute exact occupancy, ESDF, connectivity, and graph-topology targets."""

    if truncation <= 0:
        raise ValueError("ESDF truncation must be positive")
    occupied = _validate_volume(occupancy, "occupancy")
    unknown = (
        np.zeros_like(occupied)
        if unknown_mask is None
        else _validate_volume(unknown_mask, "unknown_mask")
    )
    if unknown.shape != occupied.shape:
        raise ValueError("occupancy and unknown_mask shapes must match")
    valid = ~unknown
    occupied = occupied & valid
    free = valid & ~occupied

    free_neighbors = _neighbor_count(free)
    boundary = occupied & (free_neighbors > 0)
    if not occupied.any():
        signed_distance = np.full(occupied.shape, truncation, dtype=np.float32)
    elif not free.any():
        signed_distance = np.full(occupied.shape, -truncation, dtype=np.float32)
    else:
        outside = ndimage.distance_transform_edt(~occupied)
        inside = ndimage.distance_transform_edt(occupied)
        signed_distance = np.clip(outside - inside, -truncation, truncation).astype(np.float32)
    signed_distance[~valid] = 0.0

    labels, free_components = ndimage.label(free, structure=SIX_CONNECTED)
    degree = _neighbor_count(free) * free
    dead_ends = free & (degree <= 1)
    junctions = free & (degree >= 3)
    vertices = int(free.sum())
    edges = sum(
        int((free.take(indices=range(free.shape[axis] - 1), axis=axis) &
             free.take(indices=range(1, free.shape[axis]), axis=axis)).sum())
        for axis in range(3)
    )
    graph_cycle_rank = max(0, edges - vertices + int(free_components))

    return GeometryTargets(
        occupancy=occupied.astype(np.uint8),
        valid_mask=valid,
        boundary=boundary,
        signed_distance=signed_distance,
        free_component_labels=labels.astype(np.int32),
        free_degree=degree.astype(np.uint8),
        dead_ends=dead_ends,
        junctions=junctions,
        free_components=int(free_components),
        graph_cycle_rank=graph_cycle_rank,
    )


def geodesic_distances(targets: GeometryTargets, start: tuple[int, int, int]) -> np.ndarray:
    """Return exact six-connected shortest-path distances from one free cell."""

    shape = targets.occupancy.shape
    if any(coordinate < 0 or coordinate >= shape[index] for index, coordinate in enumerate(start)):
        raise ValueError("geodesic start is outside the volume")
    free = targets.valid_mask & ~targets.occupancy.astype(bool)
    distances = np.full(shape, np.inf, dtype=np.float32)
    if not free[start]:
        return distances
    distances[start] = 0.0
    queue: deque[tuple[int, int, int]] = deque([start])
    while queue:
        x, y, z = queue.popleft()
        next_distance = distances[x, y, z] + 1.0
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            neighbor = (x + dx, y + dy, z + dz)
            if any(value < 0 or value >= shape[index] for index, value in enumerate(neighbor)):
                continue
            if free[neighbor] and not np.isfinite(distances[neighbor]):
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return distances


def pair_targets(
    targets: GeometryTargets,
    pairs: Iterable[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return reachability and geodesic distance for cell pairs."""

    reachability: list[bool] = []
    distances: list[float] = []
    cached: dict[tuple[int, int, int], np.ndarray] = {}
    labels = targets.free_component_labels
    for start, goal in pairs:
        start_label = labels[start]
        reachable = bool(start_label > 0 and start_label == labels[goal])
        reachability.append(reachable)
        if start not in cached:
            cached[start] = geodesic_distances(targets, start)
        distances.append(float(cached[start][goal]))
    return np.asarray(reachability, dtype=bool), np.asarray(distances, dtype=np.float32)
