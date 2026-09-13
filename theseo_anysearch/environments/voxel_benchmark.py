"""Strict import and a diagnostic point-path baseline for 3D voxel benchmarks.

The upstream scenario score columns and movement rules are not assumed here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from theseo_anysearch.environments.routing_manifests import (
    ConversionRecord,
    GridFrame,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceFile,
    SourceRecord,
    SplitMember,
    validate_routing_bundle,
    validate_task_endpoints,
)
from theseo_anysearch.worlds.compiler import NpySource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent

MAX_MAP_BYTES = 64 * 1024 * 1024
MAX_VOXELS = 100_000_000
NEIGHBORS_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


@dataclass(frozen=True)
class ScenarioQuery:
    row_index: int
    source_line: int
    start: tuple[int, int, int]
    goal: tuple[int, int, int]
    opaque_values: tuple[float, float, float]


@dataclass(frozen=True)
class RejectedQuery:
    source_line: int
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_process_memory_bytes() -> int:
    """Return the OS-reported process high-water mark, including native arrays."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        )
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.peak_working_set_size)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if os.uname().sysname == "Darwin" else peak * 1024)


def load_voxel_map(path: Path) -> tuple[np.ndarray, str, int]:
    """Load one strict ASCII .3dmap member into x/y/z-ordered occupancy."""

    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError("map archive must contain exactly one file")
        member = members[0]
        parts = member.filename.replace("\\", "/").split("/")
        if (
            not member.filename.endswith(".3dmap")
            or any(part in {"", ".", ".."} for part in parts)
            or member.file_size > MAX_MAP_BYTES
        ):
            raise ValueError("unsafe or oversized map archive member")
        lines = archive.read(member).decode("ascii").splitlines()
    if not lines:
        raise ValueError("empty voxel map")
    header = lines[0].split()
    if len(header) != 4 or header[0] not in {"voxel", "rev_voxel"}:
        raise ValueError("invalid voxel map header")
    try:
        dimensions = tuple(int(token) for token in header[1:])
    except ValueError as exc:
        raise ValueError("non-integer voxel map extent") from exc
    if any(value <= 0 for value in dimensions) or math.prod(dimensions) > MAX_VOXELS:
        raise ValueError("voxel map extent is invalid or too large")
    grid = np.full(dimensions, header[0] == "rev_voxel", dtype=np.uint8)
    repeated_coordinates = 0
    listed_value = int(header[0] == "voxel")
    for line_number, line in enumerate(lines[1:], start=2):
        tokens = line.split()
        if len(tokens) != 3:
            raise ValueError(f"map line {line_number}: expected exactly three axes")
        try:
            position = tuple(int(token) for token in tokens)
        except ValueError as exc:
            raise ValueError(f"map line {line_number}: non-integer coordinate") from exc
        if any(
            not 0 <= position[axis] < dimensions[axis] for axis in range(3)
        ):
            raise ValueError(f"map line {line_number}: out-of-bounds voxel")
        if grid[position] == listed_value:
            repeated_coordinates += 1
        grid[position] = listed_value
    return grid, member.filename, repeated_coordinates


def parse_scenarios(
    path: Path, map_name: str, grid: np.ndarray
) -> tuple[list[ScenarioQuery], list[RejectedQuery]]:
    """Keep every source row accounted for; reject malformed queries explicitly."""

    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 2 or lines[0].strip() != "version 2":
        raise ValueError("scenario must start with version 2")
    if lines[1].strip() != Path(map_name).name:
        raise ValueError("scenario map name does not match the ZIP member")
    accepted: list[ScenarioQuery] = []
    rejected: list[RejectedQuery] = []
    for row_index, line in enumerate(lines[2:]):
        line_number = row_index + 3
        tokens = line.split()
        if len(tokens) != 9:
            rejected.append(RejectedQuery(line_number, "expected nine columns"))
            continue
        try:
            endpoints = tuple(int(token) for token in tokens[:6])
            opaque = tuple(float(token) for token in tokens[6:])
        except ValueError:
            rejected.append(RejectedQuery(line_number, "non-numeric query value"))
            continue
        if not all(math.isfinite(value) for value in opaque):
            rejected.append(RejectedQuery(line_number, "non-finite opaque value"))
            continue
        start, goal = endpoints[:3], endpoints[3:]
        if any(
            not 0 <= endpoint[axis] < grid.shape[axis]
            for endpoint in (start, goal)
            for axis in range(3)
        ):
            rejected.append(RejectedQuery(line_number, "endpoint out of bounds"))
        elif grid[start] or grid[goal]:
            rejected.append(RejectedQuery(line_number, "occupied endpoint"))
        else:
            accepted.append(
                ScenarioQuery(row_index, line_number, start, goal, opaque)
            )
    return accepted, rejected


class _ImplicitNeighbors:
    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid
        self.expansions = 0

    def __getitem__(
        self, node: tuple[int, int, int]
    ) -> dict[tuple[int, int, int], dict[str, float]]:
        self.expansions += 1
        result = {}
        for offset in NEIGHBORS_26:
            neighbor = tuple(node[axis] + offset[axis] for axis in range(3))
            if all(0 <= neighbor[axis] < self.grid.shape[axis] for axis in range(3)):
                if not self.grid[neighbor] and self._corner_clear(node, offset):
                    result[neighbor] = {"weight": math.sqrt(sum(v * v for v in offset))}
        return result

    def _corner_clear(
        self, node: tuple[int, int, int], offset: tuple[int, int, int]
    ) -> bool:
        axes = [axis for axis in range(3) if offset[axis]]
        for count in range(1, len(axes)):
            for subset in itertools.combinations(axes, count):
                intermediate = tuple(
                    node[axis] + (offset[axis] if axis in subset else 0)
                    for axis in range(3)
                )
                if self.grid[intermediate]:
                    return False
        return True


class _ImplicitVoxelGraph:
    """Minimal graph adapter for NetworkX A* without materializing all edges."""

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid
        self._adj = _ImplicitNeighbors(grid)

    def __contains__(self, node: object) -> bool:
        if not isinstance(node, tuple) or len(node) != 3:
            return False
        if any(not isinstance(value, int) for value in node):
            return False
        return all(0 <= node[axis] < self.grid.shape[axis] for axis in range(3)) and not bool(
            self.grid[node]
        )

    @staticmethod
    def is_multigraph() -> bool:
        return False


def replay_point_path(
    grid: np.ndarray, query: ScenarioQuery, path: list[tuple[int, int, int]]
) -> float:
    """Independently check endpoint, occupancy, step size and Euclidean cost."""

    if not path or path[0] != query.start or path[-1] != query.goal:
        raise ValueError("path endpoints do not match the query")
    cost = 0.0
    for position in path:
        if any(not 0 <= position[axis] < grid.shape[axis] for axis in range(3)):
            raise ValueError("path leaves the world")
        if grid[position]:
            raise ValueError("path enters an occupied voxel")
    for current, following in zip(path, path[1:]):
        delta = tuple(following[axis] - current[axis] for axis in range(3))
        if any(abs(value) > 1 for value in delta) or delta == (0, 0, 0):
            raise ValueError("path uses a non-neighbor step")
        changed = [axis for axis in range(3) if delta[axis]]
        for subset_size in range(1, len(changed)):
            for selected in itertools.combinations(changed, subset_size):
                touched = tuple(
                    current[axis] + (delta[axis] if axis in selected else 0)
                    for axis in range(3)
                )
                if grid[touched]:
                    raise ValueError("path cuts an occupied corner or edge")
        cost += math.sqrt(sum(value * value for value in delta))
    return cost


def plan_point_path(
    grid: np.ndarray, query: ScenarioQuery
) -> tuple[list[tuple[int, int, int]], float, int]:
    graph = _ImplicitVoxelGraph(grid)
    path = nx.astar_path(
        graph,
        query.start,
        query.goal,
        heuristic=_voxel_distance,
        weight="weight",
    )
    return path, replay_point_path(grid, query, path), graph._adj.expansions


def _voxel_distance(
    current: tuple[int, int, int], goal: tuple[int, int, int]
) -> float:
    near, middle, far = sorted(abs(a - b) for a, b in zip(current, goal))
    return near * math.sqrt(3) + (middle - near) * math.sqrt(2) + (far - middle)


def run_diagnostic_slice(
    map_zip: Path,
    scenarios: Path,
    output_dir: Path,
    *,
    query_limit: int = 16,
    meters_per_voxel: float = 0.1,
) -> dict:
    """Run a fixed-prefix technical slice, not a published-score comparison."""

    if query_limit < 1:
        raise ValueError("query limit must be positive")
    started = time.perf_counter()
    grid, member_name, repeated_coordinates = load_voxel_map(map_zip)
    queries, rejected = parse_scenarios(scenarios, member_name, grid)
    source = SourceRecord(
        source_id=f"ShortestPathLab/{Path(member_name).stem}",
        source_url="https://bitbucket.org/shortestpathlab/benchmarks/",
        revision="source-file-hashes",
        files=(
            SourceFile(relative_path=map_zip.name, sha256=sha256_file(map_zip), role="geometry"),
            SourceFile(relative_path=scenarios.name, sha256=sha256_file(scenarios), role="task"),
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "occupancy.npy"
    if npy_path.exists():
        existing = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        if existing.shape != grid.shape or not np.array_equal(existing, grid):
            raise ValueError("existing occupancy.npy differs from source map")
    else:
        np.save(npy_path, grid, allow_pickle=False)
    occupancy_hash = sha256_file(npy_path)
    extent = WorldExtent.from_value(grid.shape)
    conversion = ConversionRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        converter_name="strict-3dmap-to-npy",
        converter_version="1",
        parameters={"member": member_name, "semantics": "occupied-is-one"},
        output_occupancy_sha256=occupancy_hash,
    )
    world = RoutingWorldRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        conversion_identity_sha256=conversion.identity_sha256,
        occupancy_sha256=occupancy_hash,
        extent=extent,
        frame=GridFrame(source_origin_m=(0, 0, 0), meters_per_voxel=meters_per_voxel),
        root_geometry_id=Path(member_name).stem,
        topology_family="industrial_plant",
    )
    tasks = [
        RoutingTaskRecord(
            world_identity_sha256=world.identity_sha256,
            provenance="upstream",
            family="point_path",
            start_storage=query.start,
            goal_storage=query.goal,
            movement_model="source_unspecified",
            source_query_id=f"line:{query.source_line}",
            source_task_sha256=sha256_file(scenarios),
        )
        for query in queries
    ]
    for task in tasks:
        validate_task_endpoints(task, world, grid)
    split = RoutingSplitRecord(
        dataset_id=f"{Path(member_name).stem}-point-path-technical-v1",
        members=(
            SplitMember(
                world_identity_sha256=world.identity_sha256,
                root_geometry_id=world.root_geometry_id,
                topology_family=world.topology_family,
                partition="test",
            ),
        ),
    )
    dataset_identity = validate_routing_bundle(
        sources=(source,), conversions=(conversion,), worlds=(world,),
        tasks=tasks, observations=(), references=(), split=split,
    )
    compiled = compile_world((NpySource(npy_path),), extent, output_dir / "world-packs")
    results = []
    for query, task in zip(queries[:query_limit], tasks[:query_limit]):
        query_started = time.perf_counter()
        try:
            path, cost, expansions = plan_point_path(grid, query)
            status = "valid_path"
        except nx.NetworkXNoPath:
            path, cost, expansions = [], None, None
            status = "no_path"
        results.append(
            {
                "source_line": query.source_line,
                "task_identity_sha256": task.identity_sha256,
                "start_storage": query.start,
                "goal_storage": query.goal,
                "source_columns_7_to_9_opaque": query.opaque_values,
                "status": status,
                "cost": cost,
                "path_length": len(path),
                "expansions": expansions,
                "elapsed_seconds": time.perf_counter() - query_started,
            }
        )
    report = {
        "status": "technical_diagnostic_not_comparable_to_published_scores",
        "search_semantics": "26-neighbor Euclidean cost, no diagonal through filled corners or edges; voxel-distance heuristic",
        "opaque_scenario_columns": "preserved in parsed queries; upstream meanings unverified",
        "source_sha256": {map_zip.name: sha256_file(map_zip), scenarios.name: sha256_file(scenarios)},
        "source_record_sha256": source.identity_sha256,
        "conversion_sha256": conversion.identity_sha256,
        "world_sha256": world.identity_sha256,
        "world_pack_identity": compiled.manifest.identity_sha256,
        "dataset_sha256": dataset_identity,
        "query_set_sha256": hashlib.sha256(
            json.dumps([task.identity_sha256 for task in tasks], separators=(",", ":")).encode()
        ).hexdigest(),
        "extent": list(grid.shape),
        "occupied_voxels": int(grid.sum()),
        "repeated_map_coordinates": repeated_coordinates,
        "accepted_queries": len(queries),
        "rejected_queries": [item.__dict__ for item in rejected],
        "evaluated_queries": results,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_memory_bytes": peak_process_memory_bytes(),
        "rights_status": source.rights.status,
    }
    report_path = output_dir / "diagnostic-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a technical 3D point-path slice")
    parser.add_argument("map_zip", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--query-limit", type=int, default=16)
    arguments = parser.parse_args()
    report = run_diagnostic_slice(
        arguments.map_zip,
        arguments.scenarios,
        arguments.output_dir,
        query_limit=arguments.query_limit,
    )
    print(json.dumps({
        "status": report["status"],
        "accepted_queries": report["accepted_queries"],
        "rejected_queries": len(report["rejected_queries"]),
        "evaluated_queries": len(report["evaluated_queries"]),
        "dataset_sha256": report["dataset_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
