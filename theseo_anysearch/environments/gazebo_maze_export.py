"""Export collision-verified, derived tasks from Gazebo's easy_maze world."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from scipy import ndimage

from theseo_anysearch.environments.aerial_gym_export import _rotation
from theseo_anysearch.environments.routing_manifests import (
    ArtifactRef,
    ConversionRecord,
    GridFrame,
    RightsRecord,
    RoutingReferenceRecord,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceFile,
    SourceRecord,
    SplitMember,
    validate_routing_bundle,
    validate_task_endpoints,
    write_sidecar,
)
from theseo_anysearch.worlds.compiler import NpySource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent

SPEC_SHA = "1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b"
UPSTREAM_SHA = "6ecf68cf659e122be58fc09c7a2af61b956c3301"
ARCHIVES = {
    "3d_maze.tar.xz": "5159e9f38b301307f71eccbe2ce101fd434398e60fd78ddd338e891fbb66deb1",
    "common_models.tar.xz": "03f46d349e46b55d4acba5c310d44205f1dab04ba8e107016dd4520551464a50",
}
ORIGIN = (-46.0, -46.0, 0.0)
EXTENT_M = (92.0, 92.0, 9.0)
VOXEL_M = 0.5
BODY_RADIUS_M = 0.25
ROOF_UNDERSIDE_M = 8.0
ROOF_THICKNESS_M = 1.0
STORAGE_AXES_IN_SOURCE = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
QUERY_POINTS_M = (
    ("corner_to_corner", (-37.25, -37.25, 1.25), (37.25, 37.25, 1.25)),
    ("west_to_east", (-37.25, 7.25, 1.25), (37.25, 7.25, 1.25)),
    ("south_to_north", (7.25, -37.25, 1.25), (7.25, 37.25, 1.25)),
    ("local_planar_control", (-37.25, -37.25, 1.25), (-33.25, -33.25, 1.25)),
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pose(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    values = (0.0,) * 6 if element is None or not element.text else tuple(float(v) for v in element.text.split())
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("SDF pose must contain six finite metric values")
    if element is not None and (element.attrib.get("relative_to") or element.attrib.get("frame")):
        raise ValueError("relative SDF pose frames are unsupported")
    return np.asarray(values[:3], dtype=float), _rotation(values[3:])


def _compose(parent: tuple[np.ndarray, np.ndarray], child: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return parent[0] + parent[1] @ child[0], parent[1] @ child[1]


@dataclass(frozen=True)
class Collision:
    name: str
    kind: str
    center_m: tuple[float, float, float]
    rotation: np.ndarray
    size_m: tuple[float, float, float] | None = None
    radius_m: float | None = None
    length_m: float | None = None

    def contains(self, points: np.ndarray, *, padding_m: float = 0.0) -> np.ndarray:
        local = (points - np.asarray(self.center_m)) @ self.rotation
        if self.kind == "box":
            return np.all(np.abs(local) <= np.asarray(self.size_m) / 2 + padding_m + 1e-10, axis=-1)
        return (np.linalg.norm(local[..., :2], axis=-1) <= self.radius_m + padding_m + 1e-10) & (
            np.abs(local[..., 2]) <= self.length_m / 2 + padding_m + 1e-10
        )

    def segment_hits(self, start: np.ndarray, end: np.ndarray, radius_m: float) -> bool:
        local_start = (start - np.asarray(self.center_m)) @ self.rotation
        local_delta = (end - start) @ self.rotation
        if self.kind == "box":
            half = np.asarray(self.size_m) / 2 + radius_m
            enter, leave = 0.0, 1.0
            for axis in range(3):
                if abs(local_delta[axis]) < 1e-12:
                    if abs(local_start[axis]) > half[axis]:
                        return False
                    continue
                a = (-half[axis] - local_start[axis]) / local_delta[axis]
                b = (half[axis] - local_start[axis]) / local_delta[axis]
                enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
                if enter > leave:
                    return False
            return True
        # The only source cylinder is vertical; find the closest planar point over
        # the segment's interval inside its vertical slab.
        if not np.allclose(self.rotation[:, 2], (0, 0, 1), atol=1e-8):
            raise ValueError("tilted SDF cylinders are unsupported")
        height = self.length_m / 2 + radius_m
        if abs(local_delta[2]) < 1e-12:
            if abs(local_start[2]) > height:
                return False
            lo, hi = 0.0, 1.0
        else:
            a = (-height - local_start[2]) / local_delta[2]
            b = (height - local_start[2]) / local_delta[2]
            lo, hi = max(0.0, min(a, b)), min(1.0, max(a, b))
            if lo > hi:
                return False
        planar = local_delta[:2]
        denominator = float(np.dot(planar, planar))
        t = lo if denominator == 0 else float(np.clip(-np.dot(local_start[:2], planar) / denominator, lo, hi))
        return bool(np.linalg.norm(local_start[:2] + t * planar) <= self.radius_m + radius_m)


class SdfArchives:
    def __init__(self, source_root: Path, *, verify_hashes: bool = True):
        self.archives = {}
        for name, expected in ARCHIVES.items():
            path = source_root / name
            if verify_hashes and _sha_file(path) != expected:
                raise ValueError(f"source archive hash differs from pinned {name}")
            archive = tarfile.open(path, "r:xz")
            try:
                seen = set()
                total = 0
                for index, member in enumerate(archive):
                    member_path = PurePosixPath(member.name)
                    if (index >= 512 or member.name in seen or member_path.is_absolute()
                            or "\\" in member.name or ":" in member.name
                            or ".." in member_path.parts
                            or member.name.rstrip("/") != member_path.as_posix()
                            or not (member.isfile() or member.isdir())):
                        raise ValueError("unsafe or excessive archive members")
                    seen.add(member.name)
                    total += member.size
                    if member.size > 2 * 1024 * 1024 or total > 32 * 1024 * 1024:
                        raise ValueError("archive decompressed size limit exceeded")
            except Exception:
                archive.close()
                self.close()
                raise
            self.archives[name] = archive
        self.used_members: dict[str, str] = {}

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()

    def xml(self, archive_name: str, member_name: str) -> ET.Element:
        archive = self.archives[archive_name]
        try:
            member = archive.getmember(member_name)
            if not member.isfile():
                raise ValueError(f"SDF member is not a file: {member_name}")
            payload = archive.extractfile(member).read()
        except KeyError as exc:
            raise ValueError(f"missing SDF dependency: {member_name}") from exc
        self.used_members[f"{archive_name}:{member_name}"] = _sha_bytes(payload)
        return ET.fromstring(payload)

    def model(self, name: str) -> ET.Element:
        if name in {"easy_maze_3d", "maze_wall", "maze_wall_high", "maze_wall_low", "maze_wall_short"}:
            archive, directory = "3d_maze.tar.xz", f"3d_maze/{name}"
        elif name in {"sun_2", "grass_plane", "home"}:
            archive, directory = "common_models.tar.xz", f"common_models/{name}"
        else:
            raise ValueError(f"unresolved model:// dependency: {name}")
        config = self.xml(archive, f"{directory}/model.config")
        sdf = config.find("sdf")
        if sdf is None or not sdf.text or "/" in sdf.text or "\\" in sdf.text:
            raise ValueError(f"invalid model.config for {name}")
        root = self.xml(archive, f"{directory}/{sdf.text}")
        if root.tag != "sdf":
            raise ValueError(f"invalid SDF model: {name}")
        return root


def _uri(include: ET.Element) -> str:
    uri = include.findtext("uri", "")
    if not uri.startswith("model://") or "/" in uri[8:]:
        raise ValueError(f"unsupported SDF include URI: {uri}")
    return uri[8:]


def _resolve_model(archives: SdfArchives, name: str, pose: tuple[np.ndarray, np.ndarray], prefix: str) -> list[Collision]:
    root = archives.model(name)
    model = root.find("model")
    if model is None:
        if len(root) == 1 and root[0].tag == "light":
            return []
        raise ValueError(f"model://{name} has no resolvable collision model")
    pose = _compose(pose, _pose(model.find("pose")))
    result = []
    for include in model.findall("include"):
        child = _uri(include)
        result.extend(_resolve_model(archives, child, _compose(pose, _pose(include.find("pose"))), f"{prefix}/{include.findtext('name') or child}"))
    for link in model.findall("link"):
        link_pose = _compose(pose, _pose(link.find("pose")))
        for collision in link.findall("collision"):
            collision_pose = _compose(link_pose, _pose(collision.find("pose")))
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"ambiguous collision geometry in {prefix}")
            shape = geometry[0]
            label = f"{prefix}/{link.get('name')}/{collision.get('name')}"
            center = tuple(float(v) for v in collision_pose[0])
            if shape.tag == "box":
                size = tuple(float(v) for v in shape.findtext("size", "").split())
                if len(size) != 3 or any(not math.isfinite(v) or v <= 0 for v in size):
                    raise ValueError(f"invalid collision box size: {label}")
                result.append(Collision(label, "box", center, collision_pose[1], size_m=size))
            elif shape.tag == "cylinder":
                radius = float(shape.findtext("radius", "nan"))
                length = float(shape.findtext("length", "nan"))
                if not all(math.isfinite(v) and v > 0 for v in (radius, length)):
                    raise ValueError(f"invalid collision cylinder: {label}")
                result.append(Collision(label, "cylinder", center, collision_pose[1], radius_m=radius, length_m=length))
            else:
                raise ValueError(f"unsupported SDF collision shape {shape.tag}: {label}")
    return result


def read_source_collisions(source_root: Path, *, verify_hashes: bool = True) -> tuple[tuple[Collision, ...], dict[str, str]]:
    archives = SdfArchives(source_root, verify_hashes=verify_hashes)
    try:
        world = archives.xml("3d_maze.tar.xz", "3d_maze/easy_maze.world")
        if world.tag != "sdf" or world.find("world") is None:
            raise ValueError("source archive lacks an SDF world")
        collisions = []
        for include in world.find("world").findall("include"):
            name = _uri(include)
            collisions.extend(_resolve_model(archives, name, _pose(include.find("pose")), name))
        if not collisions:
            raise ValueError("world has no collision geometry")
        return tuple(collisions), dict(sorted(archives.used_members.items()))
    finally:
        archives.close()


def _extent(voxel_m: float) -> tuple[int, int, int]:
    if not math.isfinite(voxel_m) or not 0.25 <= voxel_m <= 0.5:
        raise ValueError("voxel size must be between 0.25 and 0.5 m")
    counts = np.asarray(EXTENT_M) / voxel_m
    if not math.isfinite(voxel_m) or voxel_m <= 0 or not np.allclose(counts, np.rint(counts), atol=1e-9):
        raise ValueError("voxel size must divide fixed metric bounds")
    if np.prod(np.rint(counts)) > 5_000_000:
        raise ValueError("voxel count exceeds resource limit")
    return tuple(int(v) for v in np.rint(counts)[[0, 2, 1]])


def voxel_center(cell: tuple[int, int, int], voxel_m: float) -> np.ndarray:
    storage = np.asarray(cell)
    return np.asarray(ORIGIN) + (storage[..., [0, 2, 1]] + 0.5) * voxel_m


def _cell(point_m: tuple[float, float, float], voxel_m: float) -> tuple[int, int, int]:
    source_cell = np.rint((np.asarray(point_m) - np.asarray(ORIGIN)) / voxel_m - 0.5).astype(int)
    cell = source_cell[[0, 2, 1]]
    if np.any(cell < 0) or np.any(cell >= _extent(voxel_m)):
        raise ValueError("fixed query lies outside world bounds")
    return tuple(int(v) for v in cell)


def rasterize(collisions: tuple[Collision, ...], *, voxel_m: float = VOXEL_M) -> np.ndarray:
    """Mark every cell whose volume could intersect a source collision primitive."""
    source_shape = tuple(int(round(length / voxel_m)) for length in EXTENT_M)
    grid = np.zeros(source_shape, dtype=np.uint8)
    origin = np.asarray(ORIGIN)
    for collision in collisions:
        if collision.kind == "box":
            broad = np.abs(collision.rotation) @ (np.asarray(collision.size_m) / 2)
            margin = np.sum(np.abs(collision.rotation), axis=0) * voxel_m / 2
        else:
            broad = np.abs(collision.rotation) @ np.asarray((collision.radius_m, collision.radius_m, collision.length_m / 2))
            margin = np.sum(np.abs(collision.rotation), axis=0) * voxel_m / 2
        center = np.asarray(collision.center_m)
        lo = np.maximum(0, np.floor((center - broad - voxel_m - origin) / voxel_m).astype(int))
        hi = np.minimum(np.asarray(source_shape) - 1, np.ceil((center + broad + voxel_m - origin) / voxel_m).astype(int))
        if np.any(lo > hi):
            continue
        indices = np.indices(tuple(hi - lo + 1)).reshape(3, -1).T + lo
        centers = origin + (indices + 0.5) * voxel_m
        local = (centers - center) @ collision.rotation
        if collision.kind == "box":
            hit = np.all(np.abs(local) <= np.asarray(collision.size_m) / 2 + margin + 1e-10, axis=1)
        else:
            hit = (np.linalg.norm(local[:, :2], axis=1) <= collision.radius_m + math.sqrt(2) * voxel_m / 2 + 1e-10) & (
                np.abs(local[:, 2]) <= collision.length_m / 2 + margin[2] + 1e-10
            )
        grid[tuple(indices[hit].T)] = 1
    return np.transpose(grid, (0, 2, 1))


def _clearance_mask(occupied: np.ndarray, radius_m: float, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    # Distance is measured between cell centers; subtract one cell's circumsphere
    # radius so every accepted body center is conservatively clear of full cells.
    distance = ndimage.distance_transform_edt(occupied == 0) * voxel_m
    clearance = distance - math.sqrt(3) * voxel_m / 2
    passable = (occupied == 0) & (clearance > radius_m + 1e-9)
    passable[0, :, :] = passable[-1, :, :] = False
    passable[:, 0, :] = passable[:, -1, :] = False
    passable[:, :, 0] = passable[:, :, -1] = False
    return passable, clearance


def _route(blocked: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int], *, planar: bool = False) -> tuple[tuple[int, int, int], ...] | None:
    if blocked[start] or blocked[goal]:
        return None
    queue = deque([start])
    parents: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            result = []
            while cell is not None:
                result.append(cell)
                cell = parents[cell]
            return tuple(reversed(result))
        for axis in ((0, 2) if planar else (0, 1, 2)):
            for delta in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += delta
                point = tuple(neighbor)
                if 0 <= neighbor[axis] < blocked.shape[axis] and point not in parents and not blocked[point]:
                    parents[point] = cell
                    queue.append(point)
    return None


def replay_route(route: tuple[tuple[int, int, int], ...], collisions: tuple[Collision, ...], *, voxel_m: float, radius_m: float) -> None:
    if not route:
        raise ValueError("empty route")
    for a, b in zip(route, route[1:]):
        if sum(abs(x - y) for x, y in zip(a, b)) != 1:
            raise ValueError("non-axis-adjacent route segment")
        start, end = voxel_center(a, voxel_m), voxel_center(b, voxel_m)
        if any(shape.segment_hits(start, end, radius_m) for shape in collisions):
            raise ValueError("route collides with source collision geometry")
        for point in (start, end):
            if np.any(point - radius_m < ORIGIN) or np.any(point + radius_m > np.asarray(ORIGIN) + EXTENT_M):
                raise ValueError("route body crosses world bounds")


def _census(occupied: np.ndarray, passable: np.ndarray, clearance: np.ndarray) -> dict:
    structure = ndimage.generate_binary_structure(3, 1)
    _, free_components = ndimage.label(occupied == 0, structure=structure)
    labels, clear_components = ndimage.label(passable, structure=structure)
    sizes = np.bincount(labels[passable].ravel()) if np.any(passable) else np.array([], dtype=int)
    values = clearance[passable]
    return {
        "free_voxels": int(np.count_nonzero(occupied == 0)),
        "free_components_6": int(free_components),
        "body_valid_voxels": int(np.count_nonzero(passable)),
        "body_valid_components_6": int(clear_components),
        "largest_body_component_voxels": int(sizes.max()) if sizes.size else 0,
        "body_clearance_min_m": float(values.min()) if values.size else None,
        "body_clearance_median_m": float(np.median(values)) if values.size else None,
    }


def collision_parity_census(
    occupied: np.ndarray, collisions: tuple[Collision, ...], *, voxel_m: float
) -> dict[str, dict[str, int]]:
    """Compare source geometry with its conservative voxel cells by sample stratum."""
    lower = np.asarray(ORIGIN)
    upper = lower + EXTENT_M
    rng = np.random.default_rng(417)
    random_cells = rng.integers(0, np.asarray(occupied.shape), size=(10000, 3))
    samples: dict[str, list[np.ndarray]] = {
        "random_centers": [voxel_center(random_cells, voxel_m)],
        "interiors": [], "surfaces": [], "openings": [], "borders": [], "diagonals": [],
    }
    for collision in collisions:
        center = np.asarray(collision.center_m)
        samples["interiors"].append(center)
        if collision.kind == "box":
            half = np.asarray(collision.size_m) / 2
            for axis in range(3):
                direction = collision.rotation[:, axis]
                for sign in (-1, 1):
                    samples["surfaces"].append(center + direction * sign * (half[axis] - 1e-5))
                    samples["surfaces"].append(center + direction * sign * (half[axis] + 1e-5))
        else:
            for sign in (-1, 1):
                samples["surfaces"].append(center + collision.rotation[:, 0] * sign * (collision.radius_m - 1e-5))
                samples["surfaces"].append(center + collision.rotation[:, 0] * sign * (collision.radius_m + 1e-5))
    for _, start, goal in QUERY_POINTS_M:
        samples["openings"].extend((np.asarray(start), np.asarray(goal)))
        for t in np.linspace(0, 1, 33):
            samples["diagonals"].append((1 - t) * np.asarray(start) + t * np.asarray(goal))
    for axis in range(3):
        for side in (lower[axis] + 1e-5, upper[axis] - 1e-5):
            point = (lower + upper) / 2
            point[axis] = side
            samples["borders"].append(point)
    result = {}
    for name, parts in samples.items():
        points = np.concatenate([np.asarray(part).reshape(-1, 3) for part in parts])
        inside = np.all((points >= lower) & (points < upper), axis=1)
        points = points[inside]
        source_cells = np.floor((points - lower) / voxel_m).astype(int)
        cells = source_cells[:, [0, 2, 1]]
        source_hits = np.zeros(len(points), dtype=bool)
        for collision in collisions:
            source_hits |= collision.contains(points)
        voxel_hits = occupied[tuple(cells.T)].astype(bool)
        false_free = int(np.count_nonzero(source_hits & ~voxel_hits))
        if false_free:
            raise ValueError(f"source-versus-voxel parity failure in {name}")
        result[name] = {
            "samples": len(points), "source_hit_voxel_free": false_free,
            "conservative_voxel_hit_source_free": int(np.count_nonzero(voxel_hits & ~source_hits)),
        }
    return result


def over_wall_census(passable: np.ndarray, collisions: tuple[Collision, ...], *, voxel_m: float) -> dict[str, int]:
    """Check all voxel columns under 8 m source walls for a vertical bypass."""
    horizontal = np.indices((passable.shape[0], passable.shape[2])).reshape(2, -1).T
    points = np.column_stack((
        ORIGIN[0] + (horizontal[:, 0] + 0.5) * voxel_m,
        ORIGIN[1] + (horizontal[:, 1] + 0.5) * voxel_m,
        np.full(len(horizontal), 4.0),
    ))
    covered = np.zeros(len(horizontal), dtype=bool)
    wall_count = 0
    for collision in collisions:
        if collision.kind != "box" or collision.size_m[2] != 8.0:
            continue
        bottom = collision.center_m[2] - collision.size_m[2] / 2
        if not math.isclose(bottom, 0.0, abs_tol=1e-8):
            continue
        wall_count += 1
        covered |= collision.contains(points)
    vertical_columns = passable.transpose(0, 2, 1).reshape(-1, passable.shape[1])
    valid = int(np.count_nonzero(vertical_columns[covered]))
    if wall_count == 0 or valid:
        raise ValueError("roof permits a body-valid vertical bypass over a main wall")
    return {
        "main_wall_shapes_checked": wall_count,
        "main_wall_columns_checked": int(np.count_nonzero(covered)),
        "body_valid_over_wall_cells": valid,
    }


def export_maze(source_root: Path, output_dir: Path, *, voxel_m: float = VOXEL_M,
                body_radius_m: float = BODY_RADIUS_M, compile_packs: bool = True) -> dict:
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    if not math.isfinite(body_radius_m) or body_radius_m <= 0 or body_radius_m >= 0.5:
        raise ValueError("body radius must be positive and below 0.5 m")
    source_collisions, member_hashes = read_source_collisions(source_root)
    roof = Collision("derived/collision_roof", "box", (0.0, 0.0, ROOF_UNDERSIDE_M + ROOF_THICKNESS_M / 2), np.eye(3), size_m=(EXTENT_M[0], EXTENT_M[1], ROOF_THICKNESS_M))
    roofed_collisions = source_collisions + (roof,)
    open_occupied = rasterize(source_collisions, voxel_m=voxel_m)
    roofed_occupied = rasterize(roofed_collisions, voxel_m=voxel_m)
    output_dir.mkdir(parents=True)
    open_path, roof_path = output_dir / "open-top-occupancy.npy", output_dir / "roofed-occupancy.npy"
    np.save(open_path, open_occupied, allow_pickle=False)
    np.save(roof_path, roofed_occupied, allow_pickle=False)
    extent = WorldExtent(x=_extent(voxel_m)[0], y=_extent(voxel_m)[1], z=_extent(voxel_m)[2])
    packs = {
        "open-top": compile_world([NpySource(open_path)], extent, output_dir / "world-packs"),
        "roofed": compile_world([NpySource(roof_path)], extent, output_dir / "world-packs"),
    } if compile_packs else {}
    source = SourceRecord(
        source_id="engcang-gazebo-maps-easy-maze-3d",
        source_url="https://github.com/engcang/gazebo_maps",
        revision=UPSTREAM_SHA,
        files=tuple(SourceFile(relative_path=name, sha256=sha, role="geometry" if name.startswith("3d_") else "dependency") for name, sha in ARCHIVES.items()),
        rights=RightsRecord(status="reviewed", license_expression="BSD-3-Clause", allowed_uses=("evaluation",), evidence=f"https://github.com/engcang/gazebo_maps/blob/{UPSTREAM_SHA}/LICENSE; derived evaluation only; no source archive redistributed"),
    )
    source.rights.require_allowed("evaluation")
    conversions = []
    worlds = []
    for label, path, parent in (("open-top", open_path, None), ("roofed", roof_path, "parent")):
        parent_id = worlds[0].identity_sha256 if parent else None
        conversion = ConversionRecord(
            source_content_identity_sha256=source.content_identity_sha256,
            converter_name="gazebo-easy-maze-sdf-collision",
            converter_version="1",
            parameters={"variant": label, "voxel_m": voxel_m, "roof_underside_m": ROOF_UNDERSIDE_M if parent else -1.0, "sdf_member_hashes_sha256": _sha_bytes(json.dumps(member_hashes, sort_keys=True).encode())},
            output_occupancy_sha256=_sha_file(path),
            parent_world_identity_sha256=parent_id,
        )
        world = RoutingWorldRecord(
            source_content_identity_sha256=source.content_identity_sha256,
            conversion_identity_sha256=conversion.identity_sha256,
            occupancy_sha256=conversion.output_occupancy_sha256,
            extent=extent,
            frame=GridFrame(source_origin_m=ORIGIN, storage_axes_in_source=STORAGE_AXES_IN_SOURCE,
                            meters_per_voxel=voxel_m),
            root_geometry_id=f"gazebo-easy-maze-{UPSTREAM_SHA[:12]}",
            topology_family="gazebo-easy-maze-sdf",
            site_id="gazebo-easy-maze-3d",
            parent_world_identity_sha256=parent_id,
        )
        conversions.append(conversion)
        worlds.append(world)
    passable, clearance = _clearance_mask(roofed_occupied, body_radius_m, voxel_m)
    over_wall = over_wall_census(passable, source_collisions, voxel_m=voxel_m)
    blocked = ~passable
    tasks = []
    references = []
    rejections = []
    query_results = []
    for name, start_m, goal_m in QUERY_POINTS_M:
        start, goal = _cell(start_m, voxel_m), _cell(goal_m, voxel_m)
        if blocked[start] or blocked[goal]:
            rejections.append({"query": name, "reason": "endpoint_fails_body_clearance"})
            continue
        route = _route(blocked, start, goal)
        if route is None:
            rejections.append({"query": name, "reason": "no_body_valid_6_axis_route"})
            continue
        replay_route(route, roofed_collisions, voxel_m=voxel_m, radius_m=body_radius_m)
        planar = _route(blocked, start, goal, planar=True) if start[1] == goal[1] else None
        if planar is not None:
            replay_route(planar, roofed_collisions, voxel_m=voxel_m, radius_m=body_radius_m)
        task = RoutingTaskRecord(
            world_identity_sha256=worlds[1].identity_sha256,
            provenance="derived",
            family="drone_flight",
            start_storage=start,
            goal_storage=goal,
            movement_model="6-axis voxel centers; continuous swept-sphere replay",
            body_radius_m=body_radius_m,
            ceiling_source_m=ROOF_UNDERSIDE_M,
            source_query_id=name,
            derivation_reason="Fixed roofed Gazebo maze transfer query; source world has no route task",
        )
        validate_task_endpoints(task, worlds[1], roofed_occupied)
        route_path = output_dir / f"route-{name}.json"
        route_path.write_text(json.dumps(route) + "\n", encoding="utf-8")
        reference = RoutingReferenceRecord(
            task_identity_sha256=task.identity_sha256,
            claim="independently_validated",
            route_artifact=ArtifactRef(relative_path=route_path.name, sha256=_sha_file(route_path)),
            cost=(len(route) - 1) * voxel_m,
            verification_evidence="swept-sphere replay against SDF collision primitives and derived roof",
        )
        tasks.append(task)
        references.append(reference)
        query_results.append({
            "name": name, "task_identity_sha256": task.identity_sha256,
            "start_storage": start, "goal_storage": goal,
            "start_source_m": tuple(float(v) for v in voxel_center(start, voxel_m)),
            "goal_source_m": tuple(float(v) for v in voxel_center(goal, voxel_m)),
            "route_length_m": reference.cost,
            "route_cells": len(route), "planar_route_exists": planar is not None,
            "altitude_range_m": (max(cell[1] for cell in route) - min(cell[1] for cell in route)) * voxel_m,
            "minimum_grid_clearance_m": float(min(clearance[cell] for cell in route)),
            "straight_distance_m": float(np.linalg.norm(voxel_center(goal, voxel_m) - voxel_center(start, voxel_m))),
        })
    split = RoutingSplitRecord(
        dataset_id=f"gazebo-easy-maze-roofed-{UPSTREAM_SHA[:12]}",
        members=tuple(SplitMember(world_identity_sha256=world.identity_sha256, root_geometry_id=world.root_geometry_id, topology_family=world.topology_family, site_id=world.site_id, partition="test") for world in worlds),
    )
    dataset_id = validate_routing_bundle(sources=[source], conversions=conversions, worlds=worlds, tasks=tasks, observations=[], references=references, split=split)
    write_sidecar(output_dir / "source.json", source)
    for label, conversion, world in zip(("open-top", "roofed"), conversions, worlds):
        write_sidecar(output_dir / f"conversion-{label}.json", conversion)
        write_sidecar(output_dir / f"world-{label}.json", world)
    write_sidecar(output_dir / "split.json", split)
    for name, task, reference in zip((entry["name"] for entry in query_results), tasks, references):
        write_sidecar(output_dir / f"task-{name}.json", task)
        write_sidecar(output_dir / f"reference-{name}.json", reference)
    source_parity = collision_parity_census(roofed_occupied, roofed_collisions, voxel_m=voxel_m)
    report = {
        "schema_version": 1, "issue": 417, "governing_spec_sha": SPEC_SHA,
        "upstream_sha": UPSTREAM_SHA, "source_archives_sha256": ARCHIVES,
        "sdf_member_hashes": member_hashes,
        "source_collision_count": len(source_collisions),
        "source_collision_classes": {kind: sum(c.kind == kind for c in source_collisions) for kind in ("box", "cylinder")},
        "source_light_includes": ["sun_2"],
        "bounds_min_m": ORIGIN, "extent_m": EXTENT_M, "meters_per_voxel": voxel_m,
        "storage_axes_in_source": "storage x,y,z map to source x,z,y; renderer Y-up; zero-based cell centers",
        "outside_map_policy": "occupied for finite-radius routes; open-top crop is not a bounded flight task",
        "roof_underside_m": ROOF_UNDERSIDE_M, "roof_thickness_m": ROOF_THICKNESS_M,
        "body_radius_m": body_radius_m,
        "open_top_world_identity_sha256": worlds[0].identity_sha256,
        "roofed_world_identity_sha256": worlds[1].identity_sha256,
        "open_top_occupancy_sha256": conversions[0].output_occupancy_sha256,
        "roofed_occupancy_sha256": conversions[1].output_occupancy_sha256,
        "world_packs": {
            label: {"identity_sha256": pack.manifest.identity_sha256, "pack_sha256": _sha_file(pack.pack_path)}
            for label, pack in packs.items()
        },
        "converter_source_sha256": _sha_file(Path(__file__)),
        "dataset_identity_sha256": dataset_id,
        "source_vs_voxel_parity": source_parity,
        "over_wall_census": over_wall,
        "topology_census": _census(roofed_occupied, passable, clearance),
        "fixed_query_count": len(QUERY_POINTS_M), "accepted_queries": query_results,
        "rejected_queries": rejections,
        "altitude_required_query_count": sum(not row["planar_route_exists"] for row in query_results),
        "transfer_scope": "3d_altitude_required" if any(not row["planar_route_exists"] for row in query_results) else "enclosed_planar_only",
        "observation_status": "not_exported; full truth is not a sensor input",
        "rights": "BSD-3-Clause source; evaluation use reviewed; source archives and generated corpus remain outside Git",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Directory holding pinned 3d_maze.tar.xz and common_models.tar.xz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-m", type=float, default=VOXEL_M)
    parser.add_argument("--body-radius-m", type=float, default=BODY_RADIUS_M)
    args = parser.parse_args()
    print(json.dumps(export_maze(args.source, args.output, voxel_m=args.voxel_m, body_radius_m=args.body_radius_m), sort_keys=True))


if __name__ == "__main__":
    main()
