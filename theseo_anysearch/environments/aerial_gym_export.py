"""Deterministic, source-derived compact Aerial Gym collision-box exports.

This adapter does not run Isaac Gym or claim to reproduce its random rollout.
It uses pinned upstream collision URDFs and a separately identified task recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from theseo_anysearch.environments.routing_manifests import (
    ArtifactRef,
    ConversionRecord,
    GridFrame,
    RightsRecord,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingReferenceRecord,
    RoutingWorldRecord,
    SourceFile,
    SourceRecord,
    SplitMember,
    validate_routing_bundle,
    validate_task_endpoints,
    verify_artifact,
    write_sidecar,
)
from theseo_anysearch.worlds.manifest import WorldExtent

CONVERTER_VERSION = "1"
BOUNDS_MIN = (-5.0, -5.0, -3.0)
BOUNDS_MAX = (5.0, 5.0, 3.0)
OBSTACLE = "resources/models/environment_assets/objects/1_x_1_wall.urdf"
CONFIG = (
    "aerial_gym/config/env_config/env_with_lidar_nav_obstacles.py",
    "aerial_gym/config/asset_config/lidar_nav_env_config.py",
)


@dataclass(frozen=True)
class Box:
    size_m: tuple[float, float, float]
    center_m: tuple[float, float, float]
    rotation: np.ndarray

    def contains(self, points: np.ndarray) -> np.ndarray:
        local = (points - np.asarray(self.center_m)) @ self.rotation
        return np.all(np.abs(local) <= np.asarray(self.size_m) / 2 + 1e-10, axis=-1)


@dataclass(frozen=True)
class Instance:
    urdf: str
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _rotation(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _three(text: str | None, label: str) -> tuple[float, float, float]:
    if text is None:
        raise ValueError(f"missing {label}")
    values = tuple(float(item) for item in text.split())
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} requires three finite values")
    return values  # type: ignore[return-value]


def parse_collision_boxes(path: Path, instance: Instance) -> tuple[Box, ...]:
    """Resolve one rigid URDF link's box collisions, never visual geometry."""

    root = ET.parse(path).getroot()
    links = root.findall("link")
    if root.tag != "robot" or len(links) != 1 or root.findall("joint"):
        raise ValueError("only one-link rigid URDF assets are supported")
    collisions = links[0].findall("collision")
    if not collisions:
        raise ValueError("URDF has no collision bodies")
    actor_rotation = _rotation(instance.rpy_rad)
    boxes: list[Box] = []
    for collision in collisions:
        geometry = collision.find("geometry")
        if geometry is None or len(geometry) != 1 or geometry[0].tag != "box":
            raise ValueError("only explicit URDF collision boxes are supported")
        size = _three(geometry[0].get("size"), "collision box size")
        if any(value <= 0 for value in size):
            raise ValueError("collision box size must be positive")
        origin = collision.find("origin")
        offset = _three(
            origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            "collision origin",
        )
        rpy = _three(
            origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
            "collision rotation",
        )
        center = np.asarray(instance.position_m) + actor_rotation @ np.asarray(offset)
        boxes.append(
            Box(size, tuple(float(value) for value in center), actor_rotation @ _rotation(rpy))
        )
    return tuple(boxes)


def scene_instances(seed: int, layout: str) -> tuple[Instance, ...]:
    """Sample bounded poses from a pinned, intentionally derived layout recipe."""

    if isinstance(seed, bool) or seed < 0 or layout not in {"detour", "altitude"}:
        raise ValueError("seed must be nonnegative and layout detour or altitude")
    rng = np.random.default_rng(seed)
    walls = [
        Instance(f"resources/models/environment_assets/walls/{name}_wall.urdf", position)
        for name, position in (
            ("left", (0.0, 5.0, 0.0)),
            ("right", (0.0, -5.0, 0.0)),
            ("front", (5.0, 0.0, 0.0)),
            ("back", (-5.0, 0.0, 0.0)),
            ("top", (0.0, 0.0, 3.0)),
            ("bottom", (0.0, 0.0, -3.0)),
        )
    ]
    # Keep the load-bearing barrier fixed; seed only bounded side clutter.
    if layout == "detour":
        obstacles = [Instance(OBSTACLE, (0.0, 0.0, -1.0))]
    else:
        obstacles = [
            Instance(OBSTACLE, (0.0, float(y), float(z)))
            for y in np.arange(-4.5, 5.0, 1.0)
            for z in (-2.5, -1.5, -0.5, 0.5)
        ]
    side = (-3.2, 2.8) if layout == "altitude" else (-2.8, 2.8)
    for x in side:
        y = float(rng.choice((-2.25, 2.25)))
        z = float(rng.choice((-1.5, 1.5)))
        obstacles.append(Instance(OBSTACLE, (x, y, z), (0.0, 0.0, float(rng.uniform(-0.5, 0.5)))))
    return tuple(walls + obstacles)


def rasterize_boxes(boxes: tuple[Box, ...], voxel_m: float) -> np.ndarray:
    """Conservative grid-cell versus oriented-box intersection approximation."""

    if not math.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError("voxel size must be positive and finite")
    lengths = (np.asarray(BOUNDS_MAX) - np.asarray(BOUNDS_MIN)) / voxel_m
    if not np.allclose(lengths, np.rint(lengths), atol=1e-8):
        raise ValueError("voxel size must divide every scene dimension")
    shape = tuple(int(round(value)) for value in lengths)
    occupied = np.zeros(shape, dtype=np.uint8)
    origin = np.asarray(BOUNDS_MIN)
    for box in boxes:
        half = np.asarray(box.size_m) / 2
        broad_half = np.abs(box.rotation) @ half + voxel_m / 2
        lo = np.maximum(
            0,
            np.floor((np.asarray(box.center_m) - broad_half - origin) / voxel_m - 0.5)
            .astype(int),
        )
        hi = np.minimum(
            np.asarray(shape) - 1,
            np.ceil((np.asarray(box.center_m) + broad_half - origin) / voxel_m - 0.5)
            .astype(int),
        )
        if np.any(lo > hi):
            continue
        grid = np.indices(tuple(hi - lo + 1), dtype=np.float64)
        centers = origin + (grid.reshape(3, -1).T + lo + 0.5) * voxel_m
        local = (centers - np.asarray(box.center_m)) @ box.rotation
        # A voxel half-width projected onto each box-local axis is a conservative
        # support bound. Boundary cells may be occupied even if their centers are free.
        margin = np.sum(np.abs(box.rotation), axis=0) * (voxel_m / 2)
        hit = np.all(np.abs(local) <= half + margin + 1e-10, axis=1)
        slices = tuple(slice(int(lo[i]), int(hi[i] + 1)) for i in range(3))
        occupied[slices] |= hit.reshape(tuple(hi - lo + 1)).astype(np.uint8)
    return occupied


def _storage(point_m: tuple[float, float, float], voxel_m: float) -> tuple[int, int, int]:
    point = (np.asarray(point_m) - np.asarray(BOUNDS_MIN)) / voxel_m
    value = np.floor(point).astype(int)
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def _shortest_route(
    blocked: np.ndarray,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    *,
    planar: bool = False,
) -> tuple[tuple[int, int, int], ...] | None:
    if blocked[start] or blocked[goal]:
        return None
    frontier = deque([start])
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    while frontier:
        position = frontier.popleft()
        if position == goal:
            result = []
            current: tuple[int, int, int] | None = goal
            while current is not None:
                result.append(current)
                current = parent[current]
            return tuple(reversed(result))
        for axis in range(2 if planar else 3):
            for step in (-1, 1):
                neighbor = list(position)
                neighbor[axis] += step
                coordinate = tuple(neighbor)
                if (
                    0 <= coordinate[axis] < blocked.shape[axis]
                    and coordinate not in parent
                    and not blocked[coordinate]
                ):
                    parent[coordinate] = position
                    frontier.append(coordinate)
    return None


def _segment_hits_expanded_box(
    start_m: np.ndarray, goal_m: np.ndarray, box: Box, radius_m: float
) -> bool:
    """Exact line/AABB slab check in box coordinates, conservatively expanded."""

    start = (start_m - np.asarray(box.center_m)) @ box.rotation
    delta = (goal_m - start_m) @ box.rotation
    half = np.asarray(box.size_m) / 2 + radius_m
    enter, leave = 0.0, 1.0
    for axis in range(3):
        if abs(delta[axis]) <= 1e-12:
            if abs(start[axis]) <= half[axis]:
                continue
            return False
        a = (-half[axis] - start[axis]) / delta[axis]
        b = (half[axis] - start[axis]) / delta[axis]
        enter = max(enter, min(a, b))
        leave = min(leave, max(a, b))
        if enter <= leave:
            continue
        return False
    return True


def validate_continuous_route(
    route: tuple[tuple[int, int, int], ...], boxes: tuple[Box, ...],
    voxel_m: float, body_radius_m: float,
) -> None:
    """Replay every grid-center segment against original collision boxes."""

    if not route:
        raise ValueError("route witness is empty")
    origin = np.asarray(BOUNDS_MIN)
    for start, goal in zip(route, route[1:]):
        if sum(abs(a - b) for a, b in zip(start, goal)) != 1:
            raise ValueError("route contains a non-axis-adjacent segment")
        a = origin + (np.asarray(start) + 0.5) * voxel_m
        b = origin + (np.asarray(goal) + 0.5) * voxel_m
        if any(_segment_hits_expanded_box(a, b, box, body_radius_m) for box in boxes):
            raise ValueError("route collides with an expanded source collision box")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_git_revision(source_root: Path, revision: str, paths: list[str]) -> None:
    if not (source_root / ".git").exists():
        raise ValueError("upstream source root must be a Git checkout")
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--", *paths],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not verify upstream Git revision") from exc
    if head != revision or dirty:
        raise ValueError("upstream revision differs or selected source files are modified")


def export_scene(
    source_root: Path,
    output: Path,
    *,
    revision: str,
    seed: int,
    layout: str,
    voxel_m: float = 0.25,
    body_radius_m: float = 0.25,
) -> dict:
    """Write one immutable world, derived task, and provenance bundle."""

    if output.exists():
        raise FileExistsError(output)
    if not math.isfinite(body_radius_m) or body_radius_m < 0:
        raise ValueError("body radius must be finite and nonnegative")
    instances = scene_instances(seed, layout)
    paths = sorted(set(CONFIG) | {item.urdf for item in instances})
    _verify_git_revision(source_root, revision, paths)
    source_files = tuple(
        SourceFile(
            relative_path=name,
            sha256=_sha(source_root / name),
            role="geometry" if name.endswith(".urdf") else "dependency",
        )
        for name in paths
    )
    source = SourceRecord(
        source_id="aerial-gym-compact-collision",
        source_url="https://github.com/ntnu-arl/aerial_gym_simulator",
        revision=revision,
        files=source_files,
        rights=RightsRecord(),
    )
    for item in source.files:
        verify_artifact(source_root, item)
    boxes = tuple(
        box
        for item in instances
        for box in parse_collision_boxes(source_root / item.urdf, item)
    )
    occupancy = rasterize_boxes(boxes, voxel_m)
    if layout == "detour":
        start_m, goal_m = (-2.0, 0.0, -1.0), (2.0, 0.0, -1.0)
    else:
        start_m, goal_m = (-3.0, 0.0, -1.5), (3.0, 0.0, -1.5)
    start, goal = _storage(start_m, voxel_m), _storage(goal_m, voxel_m)
    # Minkowski-expand each box by a cube containing the spherical body. A grid
    # move stays inside its two endpoint cells, so free cells give a conservative
    # continuous-collision witness for this axis-aligned movement model.
    clearance_boxes = tuple(
        Box(
            tuple(value + 2 * body_radius_m for value in box.size_m),
            box.center_m,
            box.rotation,
        )
        for box in boxes
    )
    blocked = rasterize_boxes(clearance_boxes, voxel_m).astype(bool)
    route = _shortest_route(blocked, start, goal)
    planar_route = _shortest_route(blocked, start, goal, planar=True)
    distance = None if route is None else len(route) - 1
    planar_distance = None if planar_route is None else len(planar_route) - 1
    direct_cells = int(round(abs(goal[0] - start[0])))
    if (
        distance is None
        or distance <= direct_cells
        or (layout == "altitude" and planar_distance is not None)
    ):
        raise ValueError("generated task does not have its required detour/altitude topology")
    assert route is not None
    validate_continuous_route(route, boxes, voxel_m, body_radius_m)
    output.mkdir(parents=True)
    np.save(output / "occupancy.npy", occupancy, allow_pickle=False)
    occupancy_sha = _sha(output / "occupancy.npy")
    resolved = [
        {"urdf": item.urdf, "position_m": item.position_m, "rpy_rad": item.rpy_rad}
        for item in instances
    ]
    (output / "scene-instances.json").write_text(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    scene_sha = _sha(output / "scene-instances.json")
    conversion = ConversionRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        converter_name="aerial_gym_collision_box_export",
        converter_version=CONVERTER_VERSION,
        parameters={
            "seed": seed,
            "layout": layout,
            "voxel_m": voxel_m,
            "body_radius_m": body_radius_m,
            "rasterization": "conservative_local_aabb_v1",
            "scene_instances_sha256": scene_sha,
        },
        output_occupancy_sha256=occupancy_sha,
    )
    extent = WorldExtent.from_value(occupancy.shape)
    world = RoutingWorldRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        conversion_identity_sha256=conversion.identity_sha256,
        occupancy_sha256=occupancy_sha,
        extent=extent,
        frame=GridFrame(source_origin_m=BOUNDS_MIN, meters_per_voxel=voxel_m),
        root_geometry_id=f"aerial-gym-{layout}-{seed}",
        topology_family=f"aerial-gym-{layout}",
    )
    task = RoutingTaskRecord(
        world_identity_sha256=world.identity_sha256,
        provenance="derived",
        family="drone_flight",
        start_storage=start,
        goal_storage=goal,
        movement_model="6_axis_static_grid",
        body_radius_m=body_radius_m,
        derivation_reason=(
            "Fixed collision-checked compact task on pinned source URDF boxes; "
            "not an upstream Aerial Gym episode"
        ),
    )
    validate_task_endpoints(task, world, occupancy)
    (output / "route-storage.json").write_text(
        json.dumps(route, separators=(",", ":")), encoding="utf-8"
    )
    route_sha = _sha(output / "route-storage.json")
    reference = RoutingReferenceRecord(
        task_identity_sha256=task.identity_sha256,
        claim="independently_validated",
        route_artifact=ArtifactRef(relative_path="route-storage.json", sha256=route_sha),
        cost=distance * voxel_m,
        verification_evidence=(
            "Every 6-axis grid-center segment replayed against source URDF "
            "collision boxes expanded by body radius; no optimality claim "
            "outside this conservative grid"
        ),
    )
    split = RoutingSplitRecord(
        dataset_id=f"aerial-gym-compact-{layout}-{seed}-v1",
        members=(
            SplitMember(
                world_identity_sha256=world.identity_sha256,
                root_geometry_id=world.root_geometry_id,
                topology_family=world.topology_family,
                partition="test",
            ),
        ),
    )
    dataset_sha = validate_routing_bundle(
        sources=(source,), conversions=(conversion,), worlds=(world,), tasks=(task,),
        observations=(), references=(reference,), split=split,
    )
    for name, record in (
        ("source", source), ("conversion", conversion), ("world", world),
        ("task", task), ("reference", reference), ("split", split),
    ):
        write_sidecar(output / f"{name}.json", record)
    report = {
        "schema_version": 1,
        "source_revision": revision,
        "source_content_identity_sha256": source.content_identity_sha256,
        "dataset_identity_sha256": dataset_sha,
        "world_identity_sha256": world.identity_sha256,
        "task_identity_sha256": task.identity_sha256,
        "reference_identity_sha256": reference.identity_sha256,
        "occupied_cells": int(occupancy.sum()),
        "route_cells_6_axis": distance,
        "same_altitude_route_cells": planar_distance,
        "straight_cells": direct_cells,
        "collision_boxes": len(boxes),
        "scene_instances_sha256": scene_sha,
        "route_sha256": route_sha,
        "source_rights_status": source.rights.status,
        "claim": "derived_static_collision_grid_not_native_simulator",
    }
    (output / "export-report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layout", choices=("detour", "altitude"), required=True)
    parser.add_argument("--voxel-m", type=float, default=0.25)
    parser.add_argument("--body-radius-m", type=float, default=0.25)
    args = parser.parse_args()
    report = export_scene(
        args.source_root, args.output,
        revision=args.source_revision, seed=args.seed, layout=args.layout,
        voxel_m=args.voxel_m, body_radius_m=args.body_radius_m,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
