"""Materialize the frozen real-data probe banks for v2r1 P0C/P0D."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from theseo_anysearch.garden.evaluation.occupancy import heldout_occupancy_queries
from theseo_anysearch.garden.evaluation.reachability import sample_reachability_pairs
from theseo_anysearch.garden.pilots.calibration_revision import CalibrationDataset
from theseo_anysearch.garden.pilots.corpus import V2R1_PROGRAM, make_pilot_observation
from theseo_anysearch.garden.splits import GeometryDescriptor
from theseo_anysearch.garden.targets import compute_geometry_targets


_SIX_OFFSETS = np.asarray(
    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
    dtype=np.int64,
)


@dataclass(frozen=True)
class ReachabilityMetadata:
    geometry_ids: tuple[str, ...]
    distance_bins: np.ndarray
    kinds: np.ndarray


def _allocated_counts(total: int, groups: int) -> list[int]:
    quotient, remainder = divmod(total, groups)
    return [quotient + int(index < remainder) for index in range(groups)]


def _volume_summary(
    occupancy: np.ndarray, unknown: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """Compact local context available to a radius-preserving frozen encoder."""

    occupied = np.asarray(occupancy, dtype=np.float32)
    unknown = np.asarray(unknown, dtype=np.float32)
    free = 1.0 - np.maximum(occupied, unknown)
    channels = np.stack((occupied, free, unknown))
    rows: list[np.ndarray] = []
    for width in (3, 5):
        kernel = np.ones((width, width, width), dtype=np.float32)
        sums = np.stack(
            [ndimage.convolve(channel, kernel, mode="constant") for channel in channels]
        )
        rows.append(sums[:, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]].T)
    padded = np.pad(channels, ((0, 0), (1, 1), (1, 1), (1, 1)))
    shifted = coordinates + 1
    neighbours = np.stack(
        [
            padded[
                :,
                shifted[:, 0] + offset[0],
                shifted[:, 1] + offset[1],
                shifted[:, 2] + offset[2],
            ].T
            for offset in _SIX_OFFSETS
        ],
        axis=1,
    ).reshape(len(coordinates), -1)
    side = occupied.shape[0]
    normalized = coordinates.astype(np.float32) * (2.0 / (side - 1)) - 1.0
    global_values = np.tile(
        np.asarray((occupied.mean(), unknown.mean()), dtype=np.float32),
        (len(coordinates), 1),
    )
    return np.column_stack((*rows, neighbours, global_values, normalized)).astype(np.float32)


def _projection_controls(
    train_context: np.ndarray,
    evaluation_context: np.ndarray,
    train_null: np.ndarray,
    evaluation_null: np.ndarray,
    *,
    seed: int,
    dimensions: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mean = train_context.mean(axis=0, keepdims=True)
    scale = np.maximum(train_context.std(axis=0, keepdims=True), 1e-6)
    train = (train_context - mean) / scale
    evaluation = (evaluation_context - mean) / scale
    covariance = train.T @ train / max(1, len(train) - 1)
    _, vectors = np.linalg.eigh(covariance)
    width = min(dimensions, train.shape[1])
    pca = vectors[:, -width:]
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(train.shape[1], width)) / np.sqrt(train.shape[1])
    train_controls = {
        "frequency": np.ones((len(train), 1), dtype=np.float32),
        "coordinates_only": np.asarray(train_null, dtype=np.float32),
        "pca": np.asarray(train @ pca, dtype=np.float32),
        "fixed_random_projection": np.asarray(train @ random, dtype=np.float32),
    }
    evaluation_controls = {
        "frequency": np.ones((len(evaluation), 1), dtype=np.float32),
        "coordinates_only": np.asarray(evaluation_null, dtype=np.float32),
        "pca": np.asarray(evaluation @ pca, dtype=np.float32),
        "fixed_random_projection": np.asarray(evaluation @ random, dtype=np.float32),
    }
    return train_controls, evaluation_controls


def _coordinate_bank(
    descriptors: Sequence[GeometryDescriptor], *, total: int, seed: int
) -> dict[str, object]:
    counts = _allocated_counts(total, len(descriptors))
    occupancy_context: list[np.ndarray] = []
    standard_context: list[np.ndarray] = []
    null: list[np.ndarray] = []
    occupancy_targets: list[np.ndarray] = []
    boundary_targets: list[np.ndarray] = []
    clearance_targets: list[np.ndarray] = []
    geometry_ids: list[str] = []
    for index, (descriptor, count) in enumerate(zip(descriptors, counts)):
        if index % 12 == 0:
            print(f"coordinate bank: {index}/{len(descriptors)} geometries", flush=True)
        radius = 8 if index % 2 == 0 else 16
        observation = make_pilot_observation(
            descriptor,
            1,
            radius=radius,
            program=V2R1_PROGRAM,
        )
        targets = compute_geometry_targets(
            observation.occupancy,
            unknown_mask=observation.unknown_mask,
            truncation=radius * 0.25,
        )
        heldout = heldout_occupancy_queries(
            observation.occupancy,
            observation.unknown_mask,
            count=count,
            seed=seed + index,
        )
        occupancy_context.append(
            _volume_summary(
                heldout.input_occupancy,
                heldout.input_unknown,
                heldout.coordinates,
            )
        )
        standard_context.append(
            _volume_summary(
                observation.occupancy,
                observation.unknown_mask,
                heldout.coordinates,
            )
        )
        null.append(heldout.normalized_coordinates)
        occupancy_targets.append(heldout.targets)
        boundary_targets.append(targets.boundary[tuple(heldout.coordinates.T)].astype(np.float32))
        clearance_targets.append(
            (
                targets.signed_distance[tuple(heldout.coordinates.T)]
                / max(1.0, radius * 0.25)
            ).astype(np.float32)
        )
        geometry_ids.extend([descriptor.geometry_id] * count)
    return {
        "occupancy_context": np.concatenate(occupancy_context),
        "standard_context": np.concatenate(standard_context),
        "null": np.concatenate(null),
        "occupancy_targets": np.concatenate(occupancy_targets),
        "boundary_targets": np.concatenate(boundary_targets),
        "clearance_targets": np.concatenate(clearance_targets),
        "geometry_ids": tuple(geometry_ids),
    }


def _pair_features(
    occupancy: np.ndarray,
    unknown: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    perturbation_coordinates: np.ndarray,
    perturbation_occupied: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    side = occupancy.shape[0]
    start_features = _volume_summary(occupancy, unknown, starts)
    goal_features = _volume_summary(occupancy, unknown, goals)

    def apply_delta(features: np.ndarray, endpoints: np.ndarray) -> None:
        changed = perturbation_occupied >= 0
        old = np.zeros(len(features), dtype=np.float32)
        if changed.any():
            coordinates = perturbation_coordinates[changed]
            old[changed] = occupancy[tuple(coordinates.T)].astype(np.float32)
        delta = perturbation_occupied.astype(np.float32) - old
        delta[~changed] = 0
        distance = np.abs(perturbation_coordinates - endpoints)
        for radius, offset in ((1, 0), (2, 3)):
            inside = changed & (distance.max(axis=1) <= radius)
            features[inside, offset] += delta[inside]
            features[inside, offset + 1] -= delta[inside]
        for neighbour_index, offset in enumerate(_SIX_OFFSETS):
            at_neighbour = changed & np.all(
                perturbation_coordinates == endpoints + offset, axis=1
            )
            column = 6 + 3 * neighbour_index
            features[at_neighbour, column] += delta[at_neighbour]
            features[at_neighbour, column + 1] -= delta[at_neighbour]
        features[:, 24] += delta / float(side**3)

    apply_delta(start_features, starts)
    apply_delta(goal_features, goals)
    context = np.column_stack(
        (start_features + goal_features, np.abs(start_features - goal_features))
    ).astype(np.float32)
    start_normalized = starts.astype(np.float32) * (2.0 / (side - 1)) - 1.0
    goal_normalized = goals.astype(np.float32) * (2.0 / (side - 1)) - 1.0
    null = np.column_stack(
        (start_normalized + goal_normalized, np.abs(start_normalized - goal_normalized))
    ).astype(np.float32)
    return context, null


def _reachability_bank(
    descriptors: Sequence[GeometryDescriptor], *, total: int, seed: int
) -> tuple[dict[str, object], ReachabilityMetadata]:
    counts = _allocated_counts(total, len(descriptors))
    contexts: list[np.ndarray] = []
    nulls: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    bins: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    geometry_ids: list[str] = []
    for index, (descriptor, count) in enumerate(zip(descriptors, counts)):
        if index % 12 == 0:
            print(f"reachability bank: {index}/{len(descriptors)} geometries", flush=True)
        radius = 8
        observation = make_pilot_observation(
            descriptor,
            5,
            radius=radius,
            program=V2R1_PROGRAM,
        )
        targets = compute_geometry_targets(
            observation.occupancy,
            unknown_mask=observation.unknown_mask,
            truncation=radius * 0.25,
        )
        base = sample_reachability_pairs(
            targets.occupancy,
            targets.valid_mask,
            targets.free_component_labels,
            count=min(count, 60),
            seed=seed + index,
        )
        rng = np.random.default_rng(seed * 10_000 + index)
        selected = rng.choice(len(base), size=count, replace=len(base) < count)
        context, null = _pair_features(
            observation.occupancy,
            observation.unknown_mask,
            base.starts[selected],
            base.goals[selected],
            base.perturbation_coordinates[selected],
            base.perturbation_occupied[selected],
        )
        contexts.append(context)
        nulls.append(null)
        labels.append(base.reachable[selected].astype(np.float32))
        bins.append(base.distance_bin[selected])
        kinds.append(base.kind[selected])
        geometry_ids.extend([descriptor.geometry_id] * count)
    metadata = ReachabilityMetadata(
        geometry_ids=tuple(geometry_ids),
        distance_bins=np.concatenate(bins),
        kinds=np.concatenate(kinds),
    )
    return {
        "context": np.concatenate(contexts),
        "null": np.concatenate(nulls),
        "targets": np.concatenate(labels),
        "geometry_ids": metadata.geometry_ids,
    }, metadata


def materialize_v2r1_calibration_datasets(
    train_descriptors: Sequence[GeometryDescriptor],
    evaluation_descriptors: Sequence[GeometryDescriptor],
    *,
    coordinate_train_queries: int,
    coordinate_evaluation_queries: int,
    pair_train_queries: int,
    pair_evaluation_queries: int,
    seed: int,
) -> tuple[dict[str, CalibrationDataset], ReachabilityMetadata]:
    """Build the four active component datasets without fitting on calibration."""

    train_coordinate = _coordinate_bank(
        train_descriptors, total=coordinate_train_queries, seed=seed
    )
    evaluation_coordinate = _coordinate_bank(
        evaluation_descriptors, total=coordinate_evaluation_queries, seed=seed + 1_000
    )
    train_pair, _ = _reachability_bank(
        train_descriptors, total=pair_train_queries, seed=seed + 2_000
    )
    evaluation_pair, reachability_metadata = _reachability_bank(
        evaluation_descriptors, total=pair_evaluation_queries, seed=seed + 3_000
    )

    datasets: dict[str, CalibrationDataset] = {}
    coordinate_specs = {
        "occupied_iou": ("occupancy_context", "occupancy_targets"),
        "boundary_f1": ("standard_context", "boundary_targets"),
        "clearance_nmae": ("standard_context", "clearance_targets"),
    }
    for index, (component, (feature_key, target_key)) in enumerate(coordinate_specs.items()):
        train_controls, evaluation_controls = _projection_controls(
            train_coordinate[feature_key],
            evaluation_coordinate[feature_key],
            train_coordinate["null"],
            evaluation_coordinate["null"],
            seed=seed + 4_000 + index,
        )
        datasets[component] = CalibrationDataset(
            train_context=train_coordinate[feature_key],
            train_null=train_coordinate["null"],
            train_targets=train_coordinate[target_key],
            evaluation_context=evaluation_coordinate[feature_key],
            evaluation_null=evaluation_coordinate["null"],
            evaluation_targets=evaluation_coordinate[target_key],
            evaluation_geometry_ids=evaluation_coordinate["geometry_ids"],
            train_controls=train_controls,
            evaluation_controls=evaluation_controls,
            normalizer=1.0,
        )
    train_controls, evaluation_controls = _projection_controls(
        train_pair["context"],
        evaluation_pair["context"],
        train_pair["null"],
        evaluation_pair["null"],
        seed=seed + 5_000,
    )
    datasets["reachability_auprc"] = CalibrationDataset(
        train_context=train_pair["context"],
        train_null=train_pair["null"],
        train_targets=train_pair["targets"],
        evaluation_context=evaluation_pair["context"],
        evaluation_null=evaluation_pair["null"],
        evaluation_targets=evaluation_pair["targets"],
        evaluation_geometry_ids=evaluation_pair["geometry_ids"],
        train_controls=train_controls,
        evaluation_controls=evaluation_controls,
        normalizer=1.0,
    )
    return datasets, reachability_metadata


__all__ = ["ReachabilityMetadata", "materialize_v2r1_calibration_datasets"]
