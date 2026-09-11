"""Deterministic single-mesh overview generation for compiled voxel worlds."""

from __future__ import annotations

import math
import struct
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from theseo_anysearch.worlds.manifest import WorldExtent

OVERVIEW_ALGORITHM_VERSION = 1
OVERVIEW_FILE = "overview.mesh"
OVERVIEW_TARGET_AXIS_CELLS = 128
OVERVIEW_TRIANGLE_BUDGET = 10_000
MIN_OVERVIEW_COMPONENT_CELLS = 4
MIN_OVERVIEW_COMPONENT_DENSITY = 0.25
MIN_DENSE_COMPONENT_SOURCE_VOXELS = 8
_MESH_MAGIC = b"AOM1"
_MESH_HEADER = struct.Struct("<4sIII")
_DIRECTIONS = (
    ((-1, 0, 0), 0),
    ((1, 0, 0), 1),
    ((0, -1, 0), 2),
    ((0, 1, 0), 3),
    ((0, 0, -1), 4),
    ((0, 0, 1), 5),
)


@dataclass(frozen=True)
class CoarseCell:
    """Source occupancy retained for one coarse overview cell."""

    occupied_source_voxels: int
    represented_source_volume: int
    occupied_min: tuple[int, int, int] | None = None
    occupied_max: tuple[int, int, int] | None = None

    @property
    def occupied_bounds_volume(self) -> int:
        if self.occupied_min is None or self.occupied_max is None:
            return self.represented_source_volume
        return math.prod(
            self.occupied_max[axis] - self.occupied_min[axis] + 1 for axis in range(3)
        )


@dataclass(frozen=True)
class OverviewMesh:
    """One deterministic indexed mesh in zero-based storage coordinates."""

    vertices: tuple[tuple[int, int, int], ...]
    indices: tuple[int, ...]
    voxel_scale: int

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3


def aggregate_cells(
    occupied: set[tuple[int, int, int]],
    extent: WorldExtent,
    scale: int,
) -> dict[tuple[int, int, int], CoarseCell]:
    """OR-aggregate source voxels while retaining density accounting."""

    grouped: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for coordinate in sorted(occupied):
        key = tuple(value // scale for value in coordinate)
        grouped.setdefault(key, []).append(coordinate)
    limits = extent.as_tuple()
    return {
        key: CoarseCell(
            occupied_source_voxels=len(coordinates),
            represented_source_volume=math.prod(
                [min(scale, limits[axis] - key[axis] * scale) for axis in range(3)]
            ),
            occupied_min=tuple(
                min(item[axis] for item in coordinates) for axis in range(3)
            ),
            occupied_max=tuple(
                max(item[axis] for item in coordinates) for axis in range(3)
            ),
        )
        for key, coordinates in sorted(grouped.items())
    }


def aggregate_chunks(
    chunks: Mapping[tuple[int, int, int], np.ndarray],
    extent: WorldExtent,
    chunk_shape: tuple[int, int, int],
    scale: int,
) -> dict[tuple[int, int, int], CoarseCell]:
    """Aggregate chunk slices without expanding dense worlds into coordinate tuples."""

    counts: dict[tuple[int, int, int], int] = {}
    bounds: dict[
        tuple[int, int, int],
        tuple[tuple[int, int, int], tuple[int, int, int]],
    ] = {}
    for chunk_key in sorted(chunks):
        chunk = chunks[chunk_key]
        origin = tuple(chunk_key[axis] * chunk_shape[axis] for axis in range(3))
        last = tuple(origin[axis] + chunk.shape[axis] - 1 for axis in range(3))
        first_cell = tuple(origin[axis] // scale for axis in range(3))
        last_cell = tuple(last[axis] // scale for axis in range(3))
        for cx in range(first_cell[0], last_cell[0] + 1):
            for cy in range(first_cell[1], last_cell[1] + 1):
                for cz in range(first_cell[2], last_cell[2] + 1):
                    cell = (cx, cy, cz)
                    starts = tuple(
                        max(cell[axis] * scale - origin[axis], 0) for axis in range(3)
                    )
                    stops = tuple(
                        min((cell[axis] + 1) * scale - origin[axis], chunk.shape[axis])
                        for axis in range(3)
                    )
                    view = chunk[
                        starts[0] : stops[0],
                        starts[1] : stops[1],
                        starts[2] : stops[2],
                    ]
                    occupied_count = int(np.count_nonzero(view))
                    if occupied_count:
                        counts[cell] = counts.get(cell, 0) + occupied_count
                        local_min: list[int] = []
                        local_max: list[int] = []
                        for axis in range(3):
                            other_axes = tuple(
                                candidate for candidate in range(3) if candidate != axis
                            )
                            occupied_axes = np.flatnonzero(
                                np.any(view, axis=other_axes)
                            )
                            local_min.append(
                                origin[axis] + starts[axis] + int(occupied_axes[0])
                            )
                            local_max.append(
                                origin[axis] + starts[axis] + int(occupied_axes[-1])
                            )
                        previous = bounds.get(cell)
                        if previous is None:
                            bounds[cell] = (tuple(local_min), tuple(local_max))
                        else:
                            bounds[cell] = (
                                tuple(
                                    min(previous[0][axis], local_min[axis])
                                    for axis in range(3)
                                ),
                                tuple(
                                    max(previous[1][axis], local_max[axis])
                                    for axis in range(3)
                                ),
                            )
    limits = extent.as_tuple()
    return {
        key: CoarseCell(
            occupied_source_voxels=count,
            represented_source_volume=math.prod(
                [min(scale, limits[axis] - key[axis] * scale) for axis in range(3)]
            ),
            occupied_min=bounds[key][0],
            occupied_max=bounds[key][1],
        )
        for key, count in sorted(counts.items())
    }


def filter_components(
    cells: Mapping[tuple[int, int, int], CoarseCell],
) -> dict[tuple[int, int, int], CoarseCell]:
    """Remove small sparse 6-connected components in fixed scan order."""

    remaining = set(cells)
    kept: dict[tuple[int, int, int], CoarseCell] = {}
    for seed in sorted(cells):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        queue = deque([seed])
        component: list[tuple[int, int, int]] = []
        while queue:
            coordinate = queue.popleft()
            component.append(coordinate)
            for direction, _ in _DIRECTIONS:
                neighbor = tuple(
                    coordinate[axis] + direction[axis] for axis in range(3)
                )
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        occupied_count = sum(cells[key].occupied_source_voxels for key in component)
        occupied_bounds_volume = sum(
            cells[key].occupied_bounds_volume for key in component
        )
        density = occupied_count / occupied_bounds_volume
        if len(component) >= MIN_OVERVIEW_COMPONENT_CELLS or (
            occupied_count >= MIN_DENSE_COMPONENT_SOURCE_VOXELS
            and density >= MIN_OVERVIEW_COMPONENT_DENSITY
        ):
            for key in sorted(component):
                kept[key] = cells[key]
    return kept


def coarsen_cells(
    cells: Mapping[tuple[int, int, int], CoarseCell],
    extent: WorldExtent,
    source_scale: int,
) -> dict[tuple[int, int, int], CoarseCell]:
    """Aggregate one filtered candidate into the next bottom-up candidate."""

    target_scale = source_scale * 2
    counts: dict[tuple[int, int, int], int] = {}
    child_cells: dict[
        tuple[int, int, int],
        list[tuple[tuple[int, int, int], CoarseCell]],
    ] = {}
    for key in sorted(cells):
        parent = tuple(value // 2 for value in key)
        counts[parent] = counts.get(parent, 0) + cells[key].occupied_source_voxels
        child_cells.setdefault(parent, []).append((key, cells[key]))
    limits = extent.as_tuple()
    aggregated = {
        key: CoarseCell(
            occupied_source_voxels=count,
            represented_source_volume=math.prod(
                [
                    min(target_scale, limits[axis] - key[axis] * target_scale)
                    for axis in range(3)
                ]
            ),
            occupied_min=tuple(
                min(
                    child.occupied_min[axis]
                    if child.occupied_min is not None
                    else child_key[axis] * source_scale
                    for child_key, child in child_cells[key]
                )
                for axis in range(3)
            ),
            occupied_max=tuple(
                max(
                    child.occupied_max[axis]
                    if child.occupied_max is not None
                    else min(
                        (child_key[axis] + 1) * source_scale,
                        limits[axis],
                    )
                    - 1
                    for child_key, child in child_cells[key]
                )
                for axis in range(3)
            ),
        )
        for key, count in sorted(counts.items())
    }
    return filter_components(aggregated)


def _face_vertices(
    cell: tuple[int, int, int],
    face: int,
    scale: int,
    extent: WorldExtent,
) -> tuple[tuple[int, int, int], ...]:
    minimum = tuple(value * scale for value in cell)
    maximum = tuple(
        min(minimum[axis] + scale, extent.as_tuple()[axis]) for axis in range(3)
    )
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    faces = (
        ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)),
        ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
        ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),
        ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
    )
    return faces[face]


def mesh_cells(
    cells: Mapping[tuple[int, int, int], CoarseCell],
    extent: WorldExtent,
    scale: int,
) -> OverviewMesh:
    """Extract deterministic exposed quads from one filtered occupancy."""

    occupied = set(cells)
    vertices: list[tuple[int, int, int]] = []
    indices: list[int] = []
    for cell in sorted(occupied):
        for direction, face in _DIRECTIONS:
            neighbor = tuple(cell[axis] + direction[axis] for axis in range(3))
            if neighbor in occupied:
                continue
            base = len(vertices)
            vertices.extend(_face_vertices(cell, face, scale, extent))
            indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return OverviewMesh(tuple(vertices), tuple(indices), scale)


def build_overview_mesh(
    occupied: set[tuple[int, int, int]], extent: WorldExtent
) -> OverviewMesh:
    """Build the single final overview under stable internal complexity limits."""

    initial_scale = max(
        1,
        (max(extent.as_tuple()) + OVERVIEW_TARGET_AXIS_CELLS - 1)
        // OVERVIEW_TARGET_AXIS_CELLS,
    )
    cells = filter_components(aggregate_cells(occupied, extent, initial_scale))
    scale = initial_scale
    while True:
        mesh = mesh_cells(cells, extent, scale)
        if mesh.triangle_count <= OVERVIEW_TRIANGLE_BUDGET or not cells:
            return mesh
        cells = coarsen_cells(cells, extent, scale)
        scale *= 2


def build_overview_from_chunks(
    chunks: Mapping[tuple[int, int, int], np.ndarray],
    extent: WorldExtent,
    chunk_shape: tuple[int, int, int],
) -> OverviewMesh:
    """Build an overview without materializing all source occupied coordinates."""

    target_scale = max(
        1,
        (max(extent.as_tuple()) + OVERVIEW_TARGET_AXIS_CELLS - 1)
        // OVERVIEW_TARGET_AXIS_CELLS,
    )
    initial_scale = min(target_scale, max(chunk_shape))
    cells = filter_components(
        aggregate_chunks(chunks, extent, chunk_shape, initial_scale)
    )
    scale = initial_scale
    while True:
        mesh = mesh_cells(cells, extent, scale)
        if mesh.triangle_count <= OVERVIEW_TRIANGLE_BUDGET or not cells:
            return mesh
        cells = coarsen_cells(cells, extent, scale)
        scale *= 2


def build_stl_overview_mesh(
    path: Path,
    requested_scale: float,
    extent: WorldExtent,
    padding: int,
) -> OverviewMesh:
    """Transform and deterministically simplify an ASCII STL into storage space."""

    raw_vertices: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "vertex":
                raw_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    triangle_count = len(raw_vertices) // 3
    if triangle_count == 0:
        return OverviewMesh((), (), 1)
    values = np.asarray(raw_vertices[: triangle_count * 3], dtype=np.float64)
    minima = values.min(axis=0)
    max_extent = float((values.max(axis=0) - minima).max()) or 1.0
    max_span = extent.x - 2 * padding - 1
    if max_span < 1:
        raise ValueError("STL padding leaves no overview mesh interior")
    transform_scale = min(float(requested_scale), float(max_span)) / max_extent
    transformed = np.rint(padding + (values - minima) * transform_scale).astype(
        np.int64
    )
    limits = np.asarray(extent.as_tuple(), dtype=np.int64)
    transformed = np.clip(transformed, 0, limits)

    vertices: list[tuple[int, int, int]] = []
    vertex_indices: dict[tuple[int, int, int], int] = {}
    indices: list[int] = []
    for triangle_index in range(triangle_count):
        triangle = [
            tuple(int(axis) for axis in transformed[int(triangle_index) * 3 + offset])
            for offset in range(3)
        ]
        if len(set(triangle)) != 3:
            continue
        for vertex in triangle:
            if vertex not in vertex_indices:
                vertex_indices[vertex] = len(vertices)
                vertices.append(vertex)
            indices.append(vertex_indices[vertex])
    return OverviewMesh(tuple(vertices), tuple(indices), 1)


def encode_overview_mesh(mesh: OverviewMesh) -> bytes:
    """Encode a deterministic bounded indexed mesh."""

    header = _MESH_HEADER.pack(
        _MESH_MAGIC, OVERVIEW_ALGORITHM_VERSION, len(mesh.vertices), len(mesh.indices)
    )
    vertices = np.asarray(mesh.vertices, dtype="<u4").reshape((-1, 3)).tobytes()
    indices = np.asarray(mesh.indices, dtype="<u4").tobytes()
    return header + vertices + indices


def decode_overview_mesh(payload: bytes) -> OverviewMesh:
    """Decode and structurally validate an overview mesh payload."""

    if len(payload) < _MESH_HEADER.size:
        raise ValueError("overview mesh header is truncated")
    magic, version, vertex_count, index_count = _MESH_HEADER.unpack_from(payload)
    if magic != _MESH_MAGIC or version != OVERVIEW_ALGORITHM_VERSION:
        raise ValueError("overview mesh header is invalid")
    if index_count % 3:
        raise ValueError("overview mesh index count is not triangular")
    expected = _MESH_HEADER.size + vertex_count * 12 + index_count * 4
    if len(payload) != expected:
        raise ValueError("overview mesh byte length is invalid")
    vertex_end = _MESH_HEADER.size + vertex_count * 12
    vertices_array = np.frombuffer(
        payload[_MESH_HEADER.size : vertex_end], dtype="<u4"
    ).reshape((-1, 3))
    indices_array = np.frombuffer(payload[vertex_end:], dtype="<u4")
    if indices_array.size and int(indices_array.max()) >= vertex_count:
        raise ValueError("overview mesh index exceeds vertex count")
    return OverviewMesh(
        tuple(tuple(int(axis) for axis in vertex) for vertex in vertices_array),
        tuple(int(index) for index in indices_array),
        1,
    )
