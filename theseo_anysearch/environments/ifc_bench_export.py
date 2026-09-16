"""IFC-Bench MEP routing adapter: local IFC pipe/cable-carrier geometry to
routing_manifests bundles.

Converts a hash-verified, locally checked-out West Riverside Hospital IFC
file (from the IFC-Bench dataset) into conservative swept-volume voxel
occupancy and connectivity-derived ``single_pipe``/``coupled_pipes`` tasks.

Only ``IFCPIPESEGMENT``/``IFCPIPEFITTING`` (plumbing) or
``IFCCABLECARRIERSEGMENT``/``IFCCABLECARRIERFITTING`` (electrical) solids are
extracted, falling back to generic ``IFCFLOWSEGMENT``/``IFCFLOWFITTING`` only
when the discipline-specific types are entirely absent from the file. Sibling
structural, architectural, mechanical, fire and sprinkler IFC files for the
same building are not consumed by this adapter, so an accepted route is not
thereby proof that the run is unobstructed by those elements. This module
performs no network access; callers supply an already checked-out local
source directory.

Known scope limitation, confirmed against the real, pinned
west_riverside_hospital plumbing file: the whole building spans roughly
85 x 33 x 65 m, and its smallest declared pipe radius (~6.35 mm) drives
``meters_per_voxel`` down to this module's 0.01 m floor, so a single-shot
whole-building conversion needs on the order of 1.8e11 voxels -- far past
``MAX_VOXELS`` -- and ``export_discipline`` correctly refuses it rather than
silently coarsening or truncating. Converting a real building end to end
requires cropping to one storey, riser or room before calling this adapter;
that spatial-crop step is not implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement as ifc_placement
import ifcopenshell.util.system as ifc_system
import ifcopenshell.util.unit as ifc_unit
import numpy as np
from scipy import ndimage

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
    verify_artifact,
    write_sidecar,
)
from theseo_anysearch.environments.voxel_route_replay import (
    replay_six_axis_route,
    shortest_six_axis_route,
)
from theseo_anysearch.worlds.manifest import WorldExtent

SOURCE_ID = "ifc-bench-west-riverside-hospital"
SOURCE_URL = "https://huggingface.co/datasets/sylvainHellin/ifc-bench"
SOURCE_REVISION = "e1c4b0025ac42acf50f792e46a0ab33f4382e90e"  # tag v2.0.1
GOVERNING_SPEC_SHA = "f5e701b23c9c52a01f48ca7b76ed7be89a62268a"
GENERATOR_FAMILY = "ifc_bench_west_riverside_hospital_mep_v1"

ALLOWED_SCHEMAS = ("IFC2X3", "IFC4")
MAX_ENTITY_LINES = 2_000_000

# Pinned per-file hashes for the West Riverside Hospital project directory,
# audited against the model's own project-level license.txt (CC BY 3.0
# Unported), not the Hugging Face dataset card's repository-level tag.
FILE_HASHES: dict[str, str] = {
    "plumb_ifc4.ifc": "bb53f0eb8f7295e91e0fb6ec6b79a30022a2e0e248714aab6bb1a8930c4793dd",
    "elec_ifc4.ifc": "8e404c4e4c9085338383a89ae57a3a31ebe96ee406214392a1bf69a6775dcecc",
    "license.txt": "2d7d4e28831e12cc6ae47ec03560d602e139f1f73617bb2ed3dfe1782e8c97f4",
}

DISCIPLINE_FILES: dict[str, str] = {
    "plumbing": "plumb_ifc4.ifc",
    "electrical": "elec_ifc4.ifc",
}
DISCIPLINE_TYPES: dict[str, tuple[str, str]] = {
    "plumbing": ("IFCPIPESEGMENT", "IFCPIPEFITTING"),
    "electrical": ("IFCCABLECARRIERSEGMENT", "IFCCABLECARRIERFITTING"),
}
FALLBACK_TYPES: tuple[str, str] = ("IFCFLOWSEGMENT", "IFCFLOWFITTING")

MIN_VOXEL_SIZE_M = 0.01
MAX_VOXEL_SIZE_M = 0.20
MAX_VOXELS = 5_000_000
DEFAULT_BODY_RADIUS_M = 0.02
ELEVATION_TOLERANCE_M = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream(
    source_root: Path, *, discipline: str, verify_hashes: bool = True
) -> SourceRecord:
    """Fail closed on missing files, wrong hashes and unreviewed rights."""

    if discipline not in DISCIPLINE_FILES:
        raise ValueError(f"unknown discipline: {discipline!r}")
    root = source_root.resolve(strict=True)
    relevant = (DISCIPLINE_FILES[discipline], "license.txt")
    license_text = (root / "license.txt").read_text(encoding="utf-8")
    if "CC BY 3.0" not in license_text or "West Riverside Hospital" not in license_text:
        raise ValueError("pinned West Riverside Hospital license notice is missing or changed")
    files = tuple(
        SourceFile(
            relative_path=relative,
            sha256=_sha256(root / relative),
            role="geometry" if relative != "license.txt" else "dependency",
        )
        for relative in relevant
    )
    source = SourceRecord(
        source_id=SOURCE_ID,
        source_url=SOURCE_URL,
        revision=SOURCE_REVISION,
        files=files,
        rights=RightsRecord(
            status="reviewed",
            license_expression="CC-BY-3.0",
            allowed_uses=("training", "evaluation", "redistribution"),
            evidence=(
                "license.txt at ifc-bench v2.0.1 attributes West Riverside Hospital IFC "
                "Models to Solihin, W., OpenIFC Model Repository, University of Auckland, "
                "under CC BY 3.0 Unported (confirmed by Professor Robert Amor); a "
                "project-level grant, not a confirmed per-entity clearance of every "
                "embedded manufacturer catalog object or annotation"
            ),
        ),
    )
    for artifact in source.files:
        verify_artifact(root, artifact)
        if verify_hashes:
            expected = FILE_HASHES.get(artifact.relative_path)
            if expected is not None and artifact.sha256 != expected:
                raise ValueError(f"{artifact.relative_path} does not match the pinned hash")
    return source


def _check_step_header(path: Path) -> None:
    """Bound the parse to a declared IFC2X3/IFC4 schema and a finite entity count."""

    with path.open("r", encoding="ascii", errors="strict") as stream:
        header = stream.read(4096)
    if "FILE_SCHEMA" not in header:
        raise ValueError("STEP file is missing a FILE_SCHEMA declaration")
    if not any(f"'{schema}'" in header for schema in ALLOWED_SCHEMAS):
        raise ValueError(f"STEP file schema is not one of {ALLOWED_SCHEMAS}")
    entity_lines = 0
    with path.open("r", encoding="ascii", errors="strict") as stream:
        for line in stream:
            if line.startswith("#"):
                entity_lines += 1
                if entity_lines > MAX_ENTITY_LINES:
                    raise ValueError("STEP file exceeds the bounded entity-line count")


def open_verified_ifc(path: Path) -> ifcopenshell.file:
    """Open only after bounding the schema and entity count of trusted local bytes."""

    _check_step_header(path)
    return ifcopenshell.open(str(path))


def _length_scale_to_m(ifc_file: ifcopenshell.file) -> float:
    scale = ifc_unit.calculate_unit_scale(ifc_file)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("IFC file declares a non-finite or non-positive length unit scale")
    return float(scale)


def confirm_z_up(ifc_file: ifcopenshell.file, *, scale: float) -> None:
    """Cross-check declared storey elevations against resolved world placement.

    IFC's schema declares a Z-up world coordinate system, but this must be
    independently confirmed from the model's own data rather than assumed:
    IfcBuildingStorey.Elevation is the vertical measure relative to the
    building, while its resolved ObjectPlacement is composed through the
    full Site/Building placement chain and so may carry a large constant
    absolute offset (survey coordinates, an arbitrary site origin). Rather
    than comparing an absolute value, this compares elevation DIFFERENCES
    between storey pairs against their resolved-placement Z differences,
    which cancels any such constant offset while still proving that greater
    elevation means greater Z, i.e. that Z is genuinely the vertical axis.
    """

    storeys = ifc_file.by_type("IfcBuildingStorey")
    if len(storeys) < 2:
        raise ValueError(
            "IFC file needs at least two IfcBuildingStorey entities to confirm "
            "the up-axis from elevation differences"
        )
    resolved = []
    for storey in storeys:
        if storey.Elevation is None or storey.ObjectPlacement is None:
            raise ValueError(f"storey {storey.Name!r} lacks elevation or placement evidence")
        matrix = ifc_placement.get_local_placement(storey.ObjectPlacement)
        resolved.append((storey.Name, float(storey.Elevation) * scale, float(matrix[2, 3]) * scale))
    reference_name, reference_elevation, reference_z = resolved[0]
    for name, elevation_m, placement_z_m in resolved[1:]:
        elevation_delta = elevation_m - reference_elevation
        placement_delta = placement_z_m - reference_z
        if not math.isclose(elevation_delta, placement_delta, abs_tol=ELEVATION_TOLERANCE_M):
            raise ValueError(
                f"storey {name!r} elevation differs from {reference_name!r} by "
                f"{elevation_delta:.3f} m but resolved placement Z differs by "
                f"{placement_delta:.3f} m; refusing to assume a Z-up source"
            )


def _extract_typed(ifc_file: ifcopenshell.file, types: tuple[str, str]) -> tuple[object, ...]:
    segments = ifc_file.by_type(types[0])
    fittings = ifc_file.by_type(types[1])
    return tuple(segments) + tuple(fittings)


def extract_elements(
    ifc_file: ifcopenshell.file, *, discipline: str
) -> tuple[tuple[object, ...], bool]:
    """Extract discipline-specific MEP elements, falling back only when absent."""

    typed = _extract_typed(ifc_file, DISCIPLINE_TYPES[discipline])
    if typed:
        return typed, False
    fallback = _extract_typed(ifc_file, FALLBACK_TYPES)
    if not fallback:
        raise ValueError(
            f"IFC file has neither {DISCIPLINE_TYPES[discipline]} nor "
            f"{FALLBACK_TYPES} entities for discipline {discipline!r}"
        )
    return fallback, True


def build_connectivity(
    elements: tuple[object, ...]
) -> tuple[dict[str, set[str]], int]:
    """Map extracted-element GlobalIds to connected extracted-element GlobalIds.

    Connections that reach an element outside the extracted set (an
    unextracted fixture or terminal) are dropped from the graph and reported
    separately; they are not proof of a dead end in the real network.
    """

    by_id = {element.GlobalId: element for element in elements}
    graph: dict[str, set[str]] = {element.GlobalId: set() for element in elements}
    boundary_edges = 0
    for element in elements:
        for neighbor in ifc_system.get_connected_to(element) + ifc_system.get_connected_from(
            element
        ):
            neighbor_id = getattr(neighbor, "GlobalId", None)
            if neighbor_id in by_id:
                graph[element.GlobalId].add(neighbor_id)
                graph[neighbor_id].add(element.GlobalId)
            else:
                boundary_edges += 1
    return graph, boundary_edges


def _world_aabb_m(
    element: object, *, settings: "ifcopenshell.geom.settings"
) -> tuple[np.ndarray, np.ndarray]:
    """World-space AABB in meters.

    ifcopenshell.geom.create_shape returns vertices already in SI meters
    (CONVERT_BACK_UNITS defaults to False, meaning the kernel does not
    convert its internal SI output back to the file's declared unit) — this
    must NOT be multiplied by the file's length_unit_scale a second time,
    unlike a raw STEP attribute value (e.g. an IfcCircleProfileDef.Radius)
    read directly off an entity, which IS in the file's native unit.
    """

    shape = ifcopenshell.geom.create_shape(settings, element)
    verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
    if not verts.size:
        raise ValueError(f"element {element.GlobalId} has no triangulated geometry")
    return verts.min(axis=0), verts.max(axis=0)


def _smallest_cross_section_m(elements: tuple[object, ...], *, scale: float) -> float:
    """Smallest declared circular profile radius in meters, or a conservative default.

    IFC profile radii are in the file's own declared length unit (frequently
    millimeters for real BIM exports, not meters), so this must apply the
    same unit scale as every other geometric quantity in this module.
    """

    radii = []
    for element in elements:
        representation = getattr(element, "Representation", None)
        if representation is None:
            continue
        for rep in representation.Representations:
            for item in rep.Items:
                profile = getattr(item, "SweptArea", None)
                radius = getattr(profile, "Radius", None)
                if radius:
                    radii.append(float(radius) * scale)
    if not radii:
        return DEFAULT_BODY_RADIUS_M
    return min(radii)


def voxelize(
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    origin_m: np.ndarray,
    source_extent: tuple[int, int, int],
    meters_per_voxel: float,
) -> np.ndarray:
    """Mark every source-order voxel overlapping an element AABB, conservatively.

    Builds the grid in native source axis order (x, y, z); the caller
    transposes to storage order once, matching the confirmed Z-up source and
    the native replayer's vertical storage axis (see ``confirm_z_up``).
    """

    occupied = np.zeros(source_extent, dtype=np.uint8)
    for low, high in aabbs:
        low_idx = np.floor((low - origin_m) / meters_per_voxel).astype(int)
        high_idx = np.ceil((high - origin_m) / meters_per_voxel).astype(int)
        low_idx = np.clip(low_idx, 0, np.array(source_extent) - 1)
        high_idx = np.clip(high_idx, 0, np.array(source_extent) - 1)
        occupied[
            low_idx[0] : high_idx[0] + 1,
            low_idx[1] : high_idx[1] + 1,
            low_idx[2] : high_idx[2] + 1,
        ] = 1
    return occupied


# Storage axis order is (source x, source z, source y): storage axis 1 is
# vertical for the native replayer, and the source is confirmed Z-up by
# confirm_z_up before this permutation is ever applied.
STORAGE_AXES_IN_SOURCE = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)


def _source_to_storage_cell(
    point_m: np.ndarray, *, origin_m: np.ndarray, meters_per_voxel: float
) -> tuple[int, int, int]:
    source_cell = np.floor((point_m - origin_m) / meters_per_voxel).astype(int)
    storage_cell = source_cell[[0, 2, 1]]
    return tuple(int(value) for value in storage_cell)


def _leaf_endpoint_m(
    node_id: str,
    *,
    graph: dict[str, set[str]],
    aabbs: dict[str, tuple[np.ndarray, np.ndarray]],
    centers: dict[str, np.ndarray],
    meters_per_voxel: float,
    body_radius_m: float,
) -> np.ndarray:
    """A point just past the open end of an element's own extrusion axis.

    An element's AABB center sits inside its own occupied solid, so it cannot
    be used directly as a route endpoint; this instead walks to the element's
    free tip (away from the centroid of its connections) and steps far enough
    out to clear the same swept-sphere body radius the route is checked
    against (see clearance_mask). Works for a true leaf (one connection) and,
    as a fallback for a closed loop with no leaves, for any connected node.
    """

    neighbors = graph[node_id]
    reference = (
        np.mean([centers[neighbor] for neighbor in neighbors], axis=0)
        if neighbors
        else centers[node_id]
    )
    low, high = aabbs[node_id]
    away = centers[node_id] - reference
    axis = int(np.argmax(high - low))
    sign = 1.0 if away[axis] >= 0 else -1.0
    point = centers[node_id].copy()
    tip = high[axis] if sign > 0 else low[axis]
    step_m = body_radius_m + 3.0 * meters_per_voxel
    point[axis] = tip + sign * step_m
    return point


def clearance_mask(
    occupied: np.ndarray, *, body_radius_m: float, meters_per_voxel: float
) -> np.ndarray:
    """Conservative sphere-center clearance from occupied voxel cubes and bounds."""

    if not math.isfinite(body_radius_m) or body_radius_m < 0:
        raise ValueError("body radius must be a finite nonnegative number")
    free_with_solid_boundary = np.pad(occupied == 0, 1, constant_values=False)
    center_distance_voxels = ndimage.distance_transform_edt(free_with_solid_boundary)[
        1:-1, 1:-1, 1:-1
    ]
    clearance_m = (
        center_distance_voxels - math.sqrt(3) / 2 - 0.5
    ) * meters_per_voxel
    return (occupied == 0) & (clearance_m > body_radius_m)


def select_task_endpoints(
    elements: tuple[object, ...],
    graph: dict[int, set[int]],
    centers: dict[int, np.ndarray],
) -> tuple[tuple[int, int] | None, list[str]]:
    """Pick the farthest-apart pair of leaf elements in the largest component."""

    if not graph:
        return None, ["no_extracted_elements"]
    visited: set[int] = set()
    components: list[set[int]] = []
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(graph[current] - component)
        visited |= component
        components.append(component)
    largest = max(components, key=len)
    if len(largest) < 2:
        return None, ["largest_connected_component_has_fewer_than_two_elements"]
    leaves = [node for node in largest if len(graph[node]) <= 1] or list(largest)
    best_pair = None
    best_distance = -1.0
    for i, a in enumerate(leaves):
        for b in leaves[i + 1 :]:
            distance = float(np.linalg.norm(centers[a] - centers[b]))
            if distance > best_distance:
                best_distance = distance
                best_pair = (a, b)
    rejections = []
    if len(components) > 1:
        rejections.append(
            f"{len(components) - 1} disconnected component(s) excluded from task selection"
        )
    return best_pair, rejections


def export_discipline(
    source_root: Path,
    output_dir: Path,
    *,
    discipline: str,
    partition: str,
    body_radius_m: float | None = None,
    verify_hashes: bool = True,
) -> dict:
    """Export one discipline's connected MEP network as a routing bundle."""

    if discipline not in DISCIPLINE_TYPES:
        raise ValueError(f"unknown discipline: {discipline!r}")
    if partition not in {"train", "calibration", "test"}:
        raise ValueError("partition must be train, calibration or test")
    if output_dir.exists():
        raise FileExistsError("output directory already exists")

    source = verify_upstream(source_root, discipline=discipline, verify_hashes=verify_hashes)
    source.rights.require_allowed("training" if partition == "train" else "evaluation")

    ifc_path = source_root.resolve(strict=True) / DISCIPLINE_FILES[discipline]
    ifc_file = open_verified_ifc(ifc_path)
    scale = _length_scale_to_m(ifc_file)
    confirm_z_up(ifc_file, scale=scale)

    elements, fallback_used = extract_elements(ifc_file, discipline=discipline)
    graph, boundary_edges = build_connectivity(elements)

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    aabbs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for element in elements:
        aabbs[element.GlobalId] = _world_aabb_m(element, settings=settings)
    centers = {
        key: (low + high) / 2.0 for key, (low, high) in aabbs.items()
    }

    all_low = np.min([low for low, _ in aabbs.values()], axis=0)
    all_high = np.max([high for _, high in aabbs.values()], axis=0)

    smallest_radius_m = _smallest_cross_section_m(elements, scale=scale)
    meters_per_voxel = min(
        max(smallest_radius_m / 2.0, MIN_VOXEL_SIZE_M), MAX_VOXEL_SIZE_M
    )
    resolved_body_radius_m = (
        body_radius_m if body_radius_m is not None else min(smallest_radius_m, DEFAULT_BODY_RADIUS_M)
    )
    if not math.isfinite(resolved_body_radius_m) or resolved_body_radius_m < 0:
        raise ValueError("body radius must be a finite nonnegative number")

    # Wide enough that a leaf endpoint's step past its own tip (see
    # _leaf_endpoint_m, which scales with this same body radius) still clears
    # the world's own padded boundary by the same margin clearance_mask
    # requires from any occupied cell, not just from the leaf's own solid.
    endpoint_step_m = resolved_body_radius_m + 3.0 * meters_per_voxel
    margin = np.array([max(meters_per_voxel * 2, endpoint_step_m * 2.0)] * 3)
    origin_m = all_low - margin
    span = (all_high + margin) - origin_m
    source_extent = tuple(int(math.ceil(value / meters_per_voxel)) + 1 for value in span)
    extent = (source_extent[0], source_extent[2], source_extent[1])
    if math.prod(extent) > MAX_VOXELS:
        raise ValueError(
            f"extent {extent} at {meters_per_voxel} m/voxel exceeds the {MAX_VOXELS} voxel cap"
        )

    occupied_source_order = voxelize(
        list(aabbs.values()),
        origin_m=origin_m,
        source_extent=source_extent,
        meters_per_voxel=meters_per_voxel,
    )
    occupied = np.transpose(occupied_source_order, (0, 2, 1)).copy()

    output_dir.mkdir(parents=True)
    occupancy_path = output_dir / "occupancy.npy"
    np.save(occupancy_path, occupied, allow_pickle=False)
    occupancy_sha = _sha256(occupancy_path)

    conversion = ConversionRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        converter_name="ifcopenshell-mep-extraction",
        converter_version="1",
        parameters={
            "discipline": discipline,
            "fallback_used": fallback_used,
            "meters_per_voxel": meters_per_voxel,
        },
        output_occupancy_sha256=occupancy_sha,
    )
    world = RoutingWorldRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        conversion_identity_sha256=conversion.identity_sha256,
        occupancy_sha256=occupancy_sha,
        extent=WorldExtent(x=extent[0], y=extent[1], z=extent[2]),
        frame=GridFrame(
            source_origin_m=tuple(float(value) for value in origin_m),
            storage_axes_in_source=STORAGE_AXES_IN_SOURCE,
            meters_per_voxel=meters_per_voxel,
        ),
        root_geometry_id=f"{SOURCE_ID}-{discipline}",
        topology_family=GENERATOR_FAMILY,
        site_id=SOURCE_ID,
    )

    endpoint_pair, rejections = select_task_endpoints(elements, graph, centers)
    tasks: list[RoutingTaskRecord] = []
    references: list[RoutingReferenceRecord] = []
    route_rows: list[dict] = []
    if endpoint_pair is not None:
        start_center = _leaf_endpoint_m(
            endpoint_pair[0], graph=graph, aabbs=aabbs, centers=centers,
            meters_per_voxel=meters_per_voxel, body_radius_m=resolved_body_radius_m,
        )
        goal_center = _leaf_endpoint_m(
            endpoint_pair[1], graph=graph, aabbs=aabbs, centers=centers,
            meters_per_voxel=meters_per_voxel, body_radius_m=resolved_body_radius_m,
        )
        start = _source_to_storage_cell(
            start_center, origin_m=origin_m, meters_per_voxel=meters_per_voxel
        )
        goal = _source_to_storage_cell(
            goal_center, origin_m=origin_m, meters_per_voxel=meters_per_voxel
        )
        family = "single_pipe" if discipline == "plumbing" else "coupled_pipes"
        task = RoutingTaskRecord(
            world_identity_sha256=world.identity_sha256,
            provenance="derived",
            family=family,
            start_storage=start,
            goal_storage=goal,
            movement_model="6-axis voxel-center route; swept sphere checked against occupied cubes",
            body_radius_m=resolved_body_radius_m,
            derivation_reason=(
                "Farthest-apart leaf pair in the largest connected MEP network component; "
                "not a native IFC-Bench-labeled query"
            ),
        )
        in_bounds = all(0 <= start[axis] < extent[axis] for axis in range(3)) and all(
            0 <= goal[axis] < extent[axis] for axis in range(3)
        )
        passable = clearance_mask(
            occupied, body_radius_m=resolved_body_radius_m, meters_per_voxel=meters_per_voxel
        )
        if not in_bounds:
            rejections.append("selected_endpoints_fall_outside_the_padded_world_extent")
        elif occupied[start] or occupied[goal]:
            rejections.append("selected_endpoints_fall_inside_conservative_occupancy")
        elif not passable[start] or not passable[goal]:
            rejections.append("selected_endpoints_fail_body_radius_clearance")
        else:
            validate_task_endpoints(task, world, occupied)
            route = shortest_six_axis_route(passable, start, goal)
            if route is None:
                rejections.append("no_six_axis_route_between_selected_endpoints")
            else:
                replay_six_axis_route(
                    occupied, route,
                    body_radius_m=resolved_body_radius_m,
                    meters_per_voxel=meters_per_voxel,
                )
                route_path = output_dir / "route.json"
                route_path.write_text(
                    json.dumps(route, separators=(",", ":")), encoding="utf-8"
                )
                reference = RoutingReferenceRecord(
                    task_identity_sha256=task.identity_sha256,
                    claim="independently_validated",
                    route_artifact=ArtifactRef(
                        relative_path=route_path.name, sha256=_sha256(route_path)
                    ),
                    cost=(len(route) - 1) * meters_per_voxel,
                    verification_evidence=(
                        "Every six-axis centerline segment replayed as a swept sphere "
                        "against this discipline's extracted-solid occupied voxel cubes "
                        "only; not checked against unextracted structural, architectural, "
                        "mechanical, fire or sprinkler IFC content for the same building"
                    ),
                )
                tasks.append(task)
                references.append(reference)
                route_rows.append({
                    "family": family,
                    "task_identity_sha256": task.identity_sha256,
                    "reference_identity_sha256": reference.identity_sha256,
                    "route_steps": len(route) - 1,
                    "route_cost_m": reference.cost,
                })

    split = RoutingSplitRecord(
        dataset_id=f"{SOURCE_ID}-{discipline}-v1",
        members=(SplitMember(
            world_identity_sha256=world.identity_sha256,
            root_geometry_id=world.root_geometry_id,
            topology_family=world.topology_family,
            site_id=world.site_id,
            partition=partition,
        ),),
    )
    dataset_sha = validate_routing_bundle(
        sources=[source], conversions=[conversion], worlds=[world],
        tasks=tasks, observations=[], references=references, split=split,
    )
    write_sidecar(output_dir / "source.json", source)
    write_sidecar(output_dir / "conversion.json", conversion)
    write_sidecar(output_dir / "world.json", world)
    write_sidecar(output_dir / "split.json", split)
    for index, (task, reference) in enumerate(zip(tasks, references)):
        write_sidecar(output_dir / f"task-{index:02d}.json", task)
        write_sidecar(output_dir / f"reference-{index:02d}.json", reference)

    report = {
        "schema_version": 1,
        "governing_spec_sha": GOVERNING_SPEC_SHA,
        "source_revision": SOURCE_REVISION,
        "discipline": discipline,
        "partition": partition,
        "fallback_used": fallback_used,
        "extracted_element_count": len(elements),
        "boundary_connections_excluded": boundary_edges,
        "source_identity_sha256": source.content_identity_sha256,
        "world_identity_sha256": world.identity_sha256,
        "dataset_identity_sha256": dataset_sha,
        "occupancy_sha256": occupancy_sha,
        "occupied_voxels": int(np.count_nonzero(occupied)),
        "meters_per_voxel": meters_per_voxel,
        "body_radius_m": resolved_body_radius_m,
        "task_ids": [task.identity_sha256 for task in tasks],
        "reference_ids": [reference.identity_sha256 for reference in references],
        "routes": route_rows,
        "rejected_task_strata": rejections,
        "topology_family": GENERATOR_FAMILY,
        "cross_family_holdout_supported": False,
        "non_mep_clearance_checked": False,
        "observation_status": "not_exported; full truth is not sensor-visible input",
        "route_claim": "swept-sphere voxel-cube replay against this discipline's solids only",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discipline", choices=("plumbing", "electrical"), required=True)
    parser.add_argument("--partition", choices=("train", "calibration", "test"), required=True)
    parser.add_argument("--body-radius-m", type=float, default=None)
    args = parser.parse_args()
    report = export_discipline(
        args.source, args.output,
        discipline=args.discipline, partition=args.partition,
        body_radius_m=args.body_radius_m,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
