"""Leakage-resistant held-out occupancy queries for calibration revision F3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeldoutOccupancyPlan:
    """An occupancy query set and encoder input with query cells removed."""

    input_occupancy: np.ndarray
    input_unknown: np.ndarray
    coordinates: np.ndarray
    normalized_coordinates: np.ndarray
    targets: np.ndarray
    off_grid: bool
    cross_channel: bool


def heldout_occupancy_queries(
    occupancy: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    count: int,
    seed: int,
    off_grid: bool = False,
    cross_channel: bool = False,
) -> HeldoutOccupancyPlan:
    """Hide balanced query cells before encoding and retain their labels."""

    occupied = np.asarray(occupancy, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if occupied.ndim != 3 or unknown.shape != occupied.shape:
        raise ValueError("occupancy and unknown_mask must be aligned 3D arrays")
    if count < 2 or occupied.shape[0] < 3 or len(set(occupied.shape)) != 1:
        raise ValueError("held-out occupancy requires a cubic volume and count >= 2")
    known = ~unknown
    positives = np.argwhere(occupied & known)
    negatives = np.argwhere(~occupied & known)
    if not len(positives) or not len(negatives):
        raise ValueError("held-out occupancy requires known occupied and free cells")
    rng = np.random.default_rng(seed)
    positive_count = count // 2
    negative_count = count - positive_count
    selected_positive = positives[
        rng.choice(
            len(positives), size=positive_count, replace=len(positives) < positive_count
        )
    ]
    selected_negative = negatives[
        rng.choice(
            len(negatives), size=negative_count, replace=len(negatives) < negative_count
        )
    ]
    coordinates = np.concatenate((selected_positive, selected_negative))
    coordinates = coordinates[rng.permutation(len(coordinates))].astype(np.int64)
    targets = occupied[tuple(coordinates.T)].astype(np.float32)

    input_occupancy = occupied.copy()
    input_unknown = unknown.copy()
    unique_coordinates = np.unique(coordinates, axis=0)
    input_occupancy[tuple(unique_coordinates.T)] = False
    input_unknown[tuple(unique_coordinates.T)] = True
    if cross_channel:
        input_unknown |= input_occupancy
        input_occupancy.fill(False)

    side = occupied.shape[0]
    positions = coordinates.astype(np.float64)
    if off_grid:
        positions += rng.uniform(-0.45, 0.45, size=positions.shape)
        positions = np.clip(positions, 0, side - 1)
    normalized = (positions * (2.0 / (side - 1)) - 1.0).astype(np.float32)
    for value in (input_occupancy, input_unknown, coordinates, normalized, targets):
        value.setflags(write=False)
    return HeldoutOccupancyPlan(
        input_occupancy=input_occupancy,
        input_unknown=input_unknown,
        coordinates=coordinates,
        normalized_coordinates=normalized,
        targets=targets,
        off_grid=off_grid,
        cross_channel=cross_channel,
    )


def local_context_features(
    occupancy: np.ndarray,
    unknown_mask: np.ndarray,
    coordinates: np.ndarray,
    *,
    radius: int = 2,
) -> np.ndarray:
    """Extract raw local context while forcing each query center to zero."""

    occupied = np.asarray(occupancy, dtype=np.float32)
    unknown = np.asarray(unknown_mask, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.int64)
    if radius < 1 or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("context radius must be positive and coordinates shaped (N, 3)")
    known_free = 1.0 - np.maximum(occupied, unknown)
    channels = np.stack((occupied, known_free, unknown))
    padded = np.pad(
        channels, ((0, 0), (radius, radius), (radius, radius), (radius, radius))
    )
    width = 2 * radius + 1
    rows: list[np.ndarray] = []
    for coordinate in coordinates:
        patch = padded[
            :,
            coordinate[0] : coordinate[0] + width,
            coordinate[1] : coordinate[1] + width,
            coordinate[2] : coordinate[2] + width,
        ].copy()
        patch[:, radius, radius, radius] = 0
        rows.append(patch.reshape(-1))
    return np.stack(rows).astype(np.float32)


__all__ = ["HeldoutOccupancyPlan", "heldout_occupancy_queries", "local_context_features"]
