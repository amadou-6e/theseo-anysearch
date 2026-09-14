"""Leakage-checked compact rows from imported, rights-cleared routing worlds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from theseo_anysearch.environments.routing_manifests import (
    ConversionRecord,
    Partition,
    RoutingReferenceRecord,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceRecord,
    SplitMember,
    Use,
    read_sidecar,
    validate_routing_bundle,
    validate_task_endpoints,
    verify_artifact,
)
from theseo_anysearch.environments.voxel_route_replay import axis_segment_clear
from theseo_anysearch.worlds.manifest import world_contract_fingerprint
from theseo_anysearch.garden.compact import CompactEncoder
from theseo_anysearch.garden.evaluation.probes import encoder_state_sha256
from theseo_anysearch.garden.models.outputs import VoxelLevel

GOVERNING_SPEC_SHA = "1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b"
SIDE = 33
RADIUS = SIDE // 2
SENSOR_MODEL = "synthetic_26_ray_plus_nearfield_v1"
LABEL_MODEL = "full_truth_axis_segment_aabb_v1"
CANDIDATE_LENGTHS = (4, 8)
DIRECTIONS = tuple(
    tuple(sign if axis == index else 0 for axis in range(3))
    for index in range(3)
    for sign in (-1, 1)
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ImportedWorld:
    root: Path
    source: SourceRecord
    conversion: ConversionRecord
    world: RoutingWorldRecord
    task: RoutingTaskRecord
    reference: RoutingReferenceRecord
    occupancy: np.ndarray


def load_imported_world(
    root: Path, *, source_root: Path, use: Use = "training"
) -> ImportedWorld:
    """Load a single-task export, checking rights, provenance and actual bytes."""

    worlds = load_imported_worlds(root, source_root=source_root, use=use)
    if len(worlds) != 1:
        raise ValueError("single-task loader received a multi-task export")
    return worlds[0]


def load_imported_worlds(
    root: Path, *, source_root: Path, use: Use = "training"
) -> tuple[ImportedWorld, ...]:
    """Load all tasks for one imported world with one shared occupancy array."""

    root = Path(root)
    source = read_sidecar(root / "source.json", SourceRecord)
    source.rights.require_allowed(use)
    for artifact in source.files:
        verify_artifact(Path(source_root), artifact)
    conversion = read_sidecar(root / "conversion.json", ConversionRecord)
    world = read_sidecar(root / "world.json", RoutingWorldRecord)
    task_paths = [root / "task.json"] if (root / "task.json").exists() else sorted(root.glob("task-*.json"))
    reference_paths = [root / "reference.json"] if (root / "reference.json").exists() else sorted(root.glob("reference-*.json"))
    if not task_paths or len(task_paths) != len(reference_paths):
        raise ValueError("imported task and reference sidecars must match")
    tasks = tuple(read_sidecar(path, RoutingTaskRecord) for path in task_paths)
    references = tuple(read_sidecar(path, RoutingReferenceRecord) for path in reference_paths)
    original_split = read_sidecar(root / "split.json", RoutingSplitRecord)
    occupancy_path = root / "occupancy.npy"
    if _sha(occupancy_path) != world.occupancy_sha256:
        raise ValueError("imported occupancy bytes differ from the world sidecar")
    occupancy = np.load(occupancy_path, allow_pickle=False)
    for task, reference in zip(tasks, references):
        validate_task_endpoints(task, world, occupancy)
        if reference.task_identity_sha256 != task.identity_sha256:
            raise ValueError("reference does not match its numbered task")
        if reference.claim not in {"independently_validated", "certified_optimal"}:
            raise ValueError("compact routing rows require an independently checked route")
        if reference.route_artifact is None:
            raise ValueError("imported route artifact is missing")
        verify_artifact(root, reference.route_artifact)
    validate_routing_bundle(
        sources=(source,), conversions=(conversion,), worlds=(world,), tasks=tasks,
        observations=(), references=references, split=original_split,
    )
    return tuple(
        ImportedWorld(root, source, conversion, world, task, reference, occupancy)
        for task, reference in zip(tasks, references)
    )


def _crop(truth: np.ndarray, center: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return full-truth crop and an in-world mask; outside is blocked for labels."""

    result = np.ones((SIDE, SIDE, SIDE), dtype=np.uint8)
    inside = np.zeros_like(result, dtype=bool)
    source = []
    target = []
    for axis, coordinate in enumerate(center):
        low = max(0, coordinate - RADIUS)
        high = min(truth.shape[axis], coordinate + RADIUS + 1)
        source.append(slice(low, high))
        target.append(slice(low - coordinate + RADIUS, high - coordinate + RADIUS))
    result[tuple(target)] = truth[tuple(source)]
    inside[tuple(target)] = True
    return result, inside


def synthetic_observation(
    truth: np.ndarray, inside: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic partial observation; never reveal truth behind a ray hit."""

    if truth.shape != (SIDE,) * 3 or inside.shape != truth.shape:
        raise ValueError("expected aligned 33-cubed truth and in-world masks")
    visible = np.zeros_like(inside, dtype=bool)
    x, y, z = np.ogrid[-RADIUS : RADIUS + 1, -RADIUS : RADIUS + 1, -RADIUS : RADIUS + 1]
    visible |= (x * x + y * y + z * z <= 4) & inside
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                for step in range(1, RADIUS + 1):
                    cell = (RADIUS + dx * step, RADIUS + dy * step, RADIUS + dz * step)
                    if not inside[cell]:
                        break
                    visible[cell] = True
                    if truth[cell]:
                        break
    occupied = visible & truth.astype(bool)
    free = visible & ~truth.astype(bool)
    return occupied, free, ~visible


@dataclass(frozen=True)
class RoutingRow:
    row_id: str
    observation_id: str
    world_id: str
    task_id: str
    root_geometry_id: str
    topology_family: str
    partition: Partition
    anchor_storage: tuple[int, int, int]
    candidate_offset: tuple[int, int, int]
    traversable: bool
    raw_grid: np.ndarray  # observed occupied, observed free, unknown


@dataclass(frozen=True)
class PreparedRoutingDataset:
    dataset_identity_sha256: str
    query_identity_sha256: str
    split: RoutingSplitRecord
    test_topology_families: tuple[str, ...]
    rows: tuple[RoutingRow, ...]


def prepare_routing_rows(
    imported: Sequence[ImportedWorld],
    *,
    dataset_id: str,
    partitions: Mapping[str, Partition],
    test_topology_families: Sequence[str] = (),
) -> PreparedRoutingDataset:
    """Build fresh, deterministic 33-cube observations and short-path targets."""

    if not imported:
        raise ValueError("at least one imported world is required")
    for item in imported:
        item.source.rights.require_allowed("training")
        if _sha(item.root / "occupancy.npy") != item.world.occupancy_sha256:
            raise ValueError("imported occupancy changed after loading")
    if set(partitions) != {item.world.root_geometry_id for item in imported}:
        raise ValueError("partition assignment must name every root geometry exactly once")
    families = {item.world.topology_family for item in imported}
    held_out = set(test_topology_families)
    if not held_out <= families:
        raise ValueError("unknown test topology family")
    if len(families) > 1 and not held_out:
        raise ValueError("multiple topology families require an explicit test-family holdout")
    if held_out and (held_out == families or any(
        partitions[item.world.root_geometry_id] != "test"
        for item in imported if item.world.topology_family in held_out
    )):
        raise ValueError("held-out topology families must be test-only with another family available")
    unique_worlds = {item.world.identity_sha256: item for item in imported}
    split = RoutingSplitRecord(
        dataset_id=dataset_id,
        members=tuple(
            SplitMember(
                world_identity_sha256=item.world.identity_sha256,
                root_geometry_id=item.world.root_geometry_id,
                topology_family=item.world.topology_family,
                site_id=item.world.site_id,
                partition=partitions[item.world.root_geometry_id],
            )
            for item in unique_worlds.values()
        ),
    )
    sources = {item.source.identity_sha256: item.source for item in imported}
    conversions = {item.conversion.identity_sha256: item.conversion for item in imported}
    dataset_identity = validate_routing_bundle(
        sources=tuple(sources.values()),
        conversions=tuple(conversions.values()),
        worlds=tuple(item.world for item in unique_worlds.values()),
        tasks=tuple(item.task for item in imported),
        observations=(),
        references=tuple(item.reference for item in imported),
        split=split,
    )
    rows: list[RoutingRow] = []
    for item in sorted(
        imported, key=lambda entry: (entry.world.identity_sha256, entry.task.identity_sha256)
    ):
        world_id = item.world.identity_sha256
        task_id = item.task.identity_sha256
        for anchor in dict.fromkeys((item.task.start_storage, item.task.goal_storage)):
            cropped, inside = _crop(item.occupancy, anchor)
            occupied, free, unknown = synthetic_observation(cropped, inside)
            raw = np.stack((occupied, free, unknown)).astype(np.uint8)
            observation_id = world_contract_fingerprint({
                "world": world_id, "anchor": anchor, "sensor": SENSOR_MODEL,
                "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            })
            for direction in DIRECTIONS:
                for distance in CANDIDATE_LENGTHS:
                    offset = tuple(axis * distance for axis in direction)
                    endpoint = tuple(anchor[i] + offset[i] for i in range(3))
                    traversable = axis_segment_clear(
                        item.occupancy, anchor, endpoint,
                        body_radius_voxels=item.task.body_radius_m / item.world.frame.meters_per_voxel,
                    )
                    row_id = world_contract_fingerprint({
                        "task": task_id, "observation": observation_id,
                        "offset": offset, "label_model": LABEL_MODEL,
                    })
                    rows.append(RoutingRow(
                        row_id, observation_id, world_id, task_id,
                        item.world.root_geometry_id, item.world.topology_family,
                        partitions[item.world.root_geometry_id], anchor, offset,
                        traversable, raw,
                    ))
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("duplicate task/observation/query rows")
    query_identity = world_contract_fingerprint({
        "dataset": dataset_identity,
        "sensor": SENSOR_MODEL,
        "label": LABEL_MODEL,
        "rows": [(row.row_id, row.traversable) for row in rows],
    })
    return PreparedRoutingDataset(
        dataset_identity, query_identity, split, tuple(sorted(held_out)), tuple(rows)
    )


@dataclass(frozen=True)
class MatchedControls:
    row_ids: tuple[str, ...]
    labels: np.ndarray
    candidate_offsets: np.ndarray
    raw_grid: np.ndarray
    spatial: np.ndarray
    frozen_code: np.ndarray
    shuffled_code: np.ndarray
    feature_identity_sha256: str


@torch.no_grad()
def extract_frozen_features(
    prepared: PreparedRoutingDataset,
    model: CompactEncoder,
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, str]:
    """Extract code and dense features from observed voxels, never full truth."""

    if model.joint or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("feature extraction requires an entirely frozen compact encoder")
    before = encoder_state_sha256(model)
    spatial_hash = encoder_state_sha256(model.backbone)
    model.eval().to(device)
    spatial: dict[str, np.ndarray] = {}
    codes: dict[str, np.ndarray] = {}
    unique = {row.observation_id: row for row in prepared.rows}
    for observation_id, row in sorted(unique.items()):
        observed_occupied = torch.from_numpy(row.raw_grid[0][None]).to(device=device, dtype=torch.float32)
        unknown = torch.from_numpy(row.raw_grid[2][None, None]).to(device=device, dtype=torch.bool)
        level = VoxelLevel.from_occupancy(observed_occupied, unknown_mask=unknown)
        volume = model.backbone(level, unknown).local_feature_volume
        vector = model.aggregation(volume)
        if volume.shape != (1, 8, SIDE, SIDE, SIDE) or vector.ndim != 2 or len(vector) != 1:
            raise ValueError("frozen encoder returned an incompatible compact output")
        spatial[observation_id] = volume[0].detach().cpu().numpy().copy()
        codes[observation_id] = vector[0].detach().cpu().numpy().copy()
    if encoder_state_sha256(model) != before:
        raise ValueError("feature extraction mutated the frozen encoder")
    return spatial, codes, spatial_hash, before


def bind_matched_controls(
    prepared: PreparedRoutingDataset,
    *,
    spatial_by_observation: Mapping[str, np.ndarray],
    code_by_observation: Mapping[str, np.ndarray],
    spatial_state_sha256: str,
    code_state_sha256: str,
    shuffle_seed: int,
) -> MatchedControls:
    """Join frozen features by observation ID and shuffle codes across roots only."""

    rows = prepared.rows
    for state_hash in (spatial_state_sha256, code_state_sha256):
        if len(state_hash) != 64 or any(character not in "0123456789abcdef" for character in state_hash):
            raise ValueError("feature state hashes must be lowercase SHA-256")
    observation_ids = {row.observation_id for row in rows}
    if set(spatial_by_observation) != observation_ids or set(code_by_observation) != observation_ids:
        raise ValueError("frozen features must cover exactly the prepared observations")
    def stack_features(values: Mapping[str, np.ndarray]) -> np.ndarray:
        result = np.stack([np.asarray(values[row.observation_id]) for row in rows])
        if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
            raise ValueError("features must be finite numeric arrays")
        return result

    spatial = stack_features(spatial_by_observation)
    code = stack_features(code_by_observation)
    if code.ndim != 2 or spatial.ndim < 2:
        raise ValueError("expected vector codes and spatial feature arrays")
    rng = np.random.default_rng(shuffle_seed)
    by_partition: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(rows):
        by_partition.setdefault(row.partition, {}).setdefault(row.root_geometry_id, []).append(index)
    shuffled = np.empty_like(code)
    for roots in by_partition.values():
        names = sorted(roots)
        if len(names) < 2:
            raise ValueError("shuffled-code control needs two root geometries per partition")
        offset = int(rng.integers(1, len(names)))
        for index, name in enumerate(names):
            donor_indices = roots[names[(index + offset) % len(names)]]
            for row_index in roots[name]:
                shuffled[row_index] = code[int(rng.choice(donor_indices))]
    feature_identity = world_contract_fingerprint({
        "query": prepared.query_identity_sha256,
        "spatial_state": spatial_state_sha256,
        "code_state": code_state_sha256,
        "shuffle_seed": shuffle_seed,
        "spatial_sha256": hashlib.sha256(spatial.tobytes()).hexdigest(),
        "code_sha256": hashlib.sha256(code.tobytes()).hexdigest(),
        "shuffled_sha256": hashlib.sha256(shuffled.tobytes()).hexdigest(),
    })
    return MatchedControls(
        tuple(row.row_id for row in rows),
        np.asarray([row.traversable for row in rows], dtype=np.uint8),
        np.asarray([row.candidate_offset for row in rows], dtype=np.int8),
        np.stack([row.raw_grid for row in rows]), spatial, code, shuffled,
        feature_identity,
    )


def compact_report(prepared: PreparedRoutingDataset) -> dict:
    """Return a source-free report suitable for Git; raw grids stay outside Git."""

    partitions = {name: [] for name in ("train", "calibration", "test")}
    label_counts = {name: {"clear": 0, "blocked": 0} for name in partitions}
    for member in prepared.split.members:
        partitions[member.partition].append(member.root_geometry_id)
    for row in prepared.rows:
        label_counts[row.partition]["clear" if row.traversable else "blocked"] += 1
    return {
        "schema_version": 1,
        "governing_spec_sha": GOVERNING_SPEC_SHA,
        "dataset_identity_sha256": prepared.dataset_identity_sha256,
        "query_identity_sha256": prepared.query_identity_sha256,
        "sensor_model": SENSOR_MODEL,
        "label_model": LABEL_MODEL,
        "rows": len(prepared.rows),
        "traversable": sum(row.traversable for row in prepared.rows),
        "partitions": {name: sorted(set(roots)) for name, roots in partitions.items()},
        "split_coverage": (
            "three_way" if all(partitions.values()) else "incomplete_not_fit_ready"
        ),
        "label_counts": label_counts,
        "topology_families": sorted({row.topology_family for row in prepared.rows}),
        "test_topology_families": list(prepared.test_topology_families),
        "claims": "preparation_only_no_training_or_cross_family_result",
    }


def write_prepared_dataset(output: Path, prepared: PreparedRoutingDataset) -> dict:
    """Persist observed inputs and labels only; full truth stays in source exports."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = prepared.rows
    observations = {row.observation_id: row.raw_grid for row in rows}
    observation_ids = sorted(observations)
    observation_index = {value: index for index, value in enumerate(observation_ids)}
    np.savez_compressed(
        output / "rows.npz",
        raw_grid=np.stack([observations[value] for value in observation_ids]),
        observation_index=np.asarray(
            [observation_index[row.observation_id] for row in rows], dtype=np.int32
        ),
        traversable=np.asarray([row.traversable for row in rows], dtype=np.uint8),
        candidate_offset=np.asarray([row.candidate_offset for row in rows], dtype=np.int8),
    )
    metadata = [
        {
            "row_id": row.row_id,
            "observation_id": row.observation_id,
            "world_id": row.world_id,
            "task_id": row.task_id,
            "root_geometry_id": row.root_geometry_id,
            "topology_family": row.topology_family,
            "partition": row.partition,
            "anchor_storage": row.anchor_storage,
            "candidate_offset": row.candidate_offset,
        }
        for row in rows
    ]
    (output / "rows.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    report = {
        **compact_report(prepared),
        "rows_npz_sha256": _sha(output / "rows.npz"),
        "rows_json_sha256": _sha(output / "rows.json"),
        "observations": len(observation_ids),
        "source_world_ids": sorted({row.world_id for row in rows}),
    }
    (output / "prepare-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    return report


def read_prepared_dataset(output: Path) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    """Verify a persisted corpus before #416 consumes it."""

    output = Path(output)
    report = json.loads((output / "prepare-report.json").read_text(encoding="utf-8"))
    for filename, field in (("rows.npz", "rows_npz_sha256"), ("rows.json", "rows_json_sha256")):
        if _sha(output / filename) != report[field]:
            raise ValueError(f"prepared {filename} does not match its report")
    metadata = json.loads((output / "rows.json").read_text(encoding="utf-8"))
    with np.load(output / "rows.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"raw_grid", "observation_index", "traversable", "candidate_offset"}:
            raise ValueError("prepared arrays have an unexpected schema")
        arrays = {name: saved[name].copy() for name in saved.files}
    rows = len(metadata)
    raw = arrays["raw_grid"]
    labels = arrays["traversable"]
    offsets = arrays["candidate_offset"]
    observation_index = arrays["observation_index"]
    if (
        rows != report["rows"]
        or raw.shape != (report["observations"], 3, SIDE, SIDE, SIDE)
        or observation_index.shape != (rows,)
        or observation_index.dtype.kind not in "iu"
        or np.any(observation_index < 0)
        or np.any(observation_index >= len(raw))
        or labels.shape != (rows,)
        or offsets.shape != (rows, 3)
        or not np.isin(raw, (0, 1)).all()
        or not np.all(raw.sum(axis=1) == 1)
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("prepared arrays have invalid shapes or observation masks")
    if len({row["row_id"] for row in metadata}) != rows:
        raise ValueError("prepared rows contain duplicate IDs")
    observed_ids = sorted({row["observation_id"] for row in metadata})
    if len(observed_ids) != len(raw) or any(
        observed_ids[observation_index[index]] != row["observation_id"]
        for index, row in enumerate(metadata)
    ):
        raise ValueError("prepared observation indices disagree with row IDs")
    for index, row in enumerate(metadata):
        if tuple(row["candidate_offset"]) != tuple(offsets[index]):
            raise ValueError("prepared query offsets disagree with metadata")
    query_identity = world_contract_fingerprint({
        "dataset": report["dataset_identity_sha256"],
        "sensor": report["sensor_model"],
        "label": report["label_model"],
        "rows": [(row["row_id"], bool(labels[index])) for index, row in enumerate(metadata)],
    })
    if query_identity != report["query_identity_sha256"]:
        raise ValueError("prepared query identity does not match rows and labels")
    return report, metadata, arrays
