"""Independent validation of provider output against the shared routing contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from theseo_anysearch.environments.routing_manifests import (
    ConversionRecord,
    RoutingReferenceRecord,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceRecord,
    read_sidecar,
    validate_routing_bundle,
    validate_task_endpoints,
    verify_artifact,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    source: SourceRecord
    conversion: ConversionRecord
    world: RoutingWorldRecord
    tasks: tuple[RoutingTaskRecord, ...]
    references: tuple[RoutingReferenceRecord, ...]
    split: RoutingSplitRecord
    dataset_identity_sha256: str
    occupancy: np.ndarray


def _segment_clear(grid: np.ndarray, a: tuple[int, int, int], b: tuple[int, int, int], radius: float) -> bool:
    """Exact axis-segment distance to nearby occupied voxel cubes and solid bounds."""

    shape = grid.shape
    if any(
        min(a[i], b[i]) - radius <= -0.5 or max(a[i], b[i]) + radius >= shape[i] - 0.5
        for i in range(3)
    ):
        return False
    axis = next(i for i in range(3) if a[i] != b[i])
    margin = math.ceil(radius + 0.5)
    lower = [max(0, min(a[i], b[i]) - margin) for i in range(3)]
    upper = [min(shape[i], max(a[i], b[i]) + margin + 1) for i in range(3)]
    region = grid[tuple(slice(x, y) for x, y in zip(lower, upper))]
    occupied = np.argwhere(region != 0)
    if not len(occupied):
        return True
    occupied += np.asarray(lower)
    distances = np.zeros((len(occupied), 3), dtype=np.float64)
    for i in range(3):
        if i == axis:
            low, high = sorted((a[i], b[i]))
            distances[:, i] = np.maximum.reduce((
                occupied[:, i] - 0.5 - high,
                low - occupied[:, i] - 0.5,
                np.zeros(len(occupied)),
            ))
        else:
            distances[:, i] = np.maximum(np.abs(occupied[:, i] - a[i]) - 0.5, 0)
    return bool(np.all(np.sum(distances**2, axis=1) > radius**2))


def _read_route(root: Path, reference: RoutingReferenceRecord) -> tuple[tuple[int, int, int], ...]:
    artifact = reference.route_artifact
    if artifact is None:
        raise ValueError("reference has no route artifact")
    verify_artifact(root, artifact)
    raw = json.loads((root / artifact.relative_path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < 2 or any(
        not isinstance(cell, list) or len(cell) != 3 or any(type(v) is not int for v in cell)
        for cell in raw
    ):
        raise ValueError("reference route must contain integer voxel coordinates")
    return tuple(tuple(cell) for cell in raw)  # type: ignore[return-value]


def load_bundle(root: Path, *, use: str | None = None) -> VerifiedBundle:
    root = root.resolve(strict=True)
    source = read_sidecar(root / "source.json", SourceRecord)
    for source_file in source.files:
        if (root / source_file.relative_path).exists():
            verify_artifact(root, source_file)
    conversion = read_sidecar(root / "conversion.json", ConversionRecord)
    world = read_sidecar(root / "world.json", RoutingWorldRecord)
    split = read_sidecar(root / "split.json", RoutingSplitRecord)
    tasks = tuple(read_sidecar(path, RoutingTaskRecord) for path in sorted(root.glob("task-*.json")))
    references = tuple(
        read_sidecar(path, RoutingReferenceRecord) for path in sorted(root.glob("reference-*.json"))
    )
    if not tasks or len(tasks) != len(references):
        raise ValueError("verified world requires matching nonempty tasks and references")
    if use is not None:
        source.rights.require_allowed(use)  # type: ignore[arg-type]
    occupancy_path = root / "occupancy.npy"
    if sha256(occupancy_path) != world.occupancy_sha256:
        raise ValueError("occupied voxel grid differs from the world sidecar")
    occupancy = np.load(occupancy_path, allow_pickle=False)
    if occupancy.dtype != np.uint8 or occupancy.shape != world.extent.as_tuple() or np.any(occupancy > 1):
        raise ValueError("occupancy must be a complete binary uint8 grid of the declared extent")
    identity = validate_routing_bundle(
        sources=[source], conversions=[conversion], worlds=[world], tasks=tasks,
        observations=[], references=references, split=split,
    )
    reference_by_task = {item.task_identity_sha256: item for item in references}
    if len(reference_by_task) != len(tasks):
        raise ValueError("duplicate or missing task references")
    for task in tasks:
        validate_task_endpoints(task, world, occupancy)
        reference = reference_by_task.get(task.identity_sha256)
        if reference is None or reference.claim not in {"independently_validated", "certified_optimal"}:
            raise ValueError("every task needs an independently validated reference")
        route = _read_route(root, reference)
        if route[0] != task.start_storage or route[-1] != task.goal_storage:
            raise ValueError("route endpoints disagree with task")
        radius = task.body_radius_m / world.frame.meters_per_voxel
        for a, b in zip(route, route[1:]):
            if sum(abs(x - y) for x, y in zip(a, b)) != 1:
                raise ValueError("route contains a non-six-axis step")
            if not _segment_clear(occupancy, a, b, radius):
                raise ValueError("route collides with occupied voxel cubes or world bounds")
    return VerifiedBundle(root, source, conversion, world, tasks, references, split, identity, occupancy)
