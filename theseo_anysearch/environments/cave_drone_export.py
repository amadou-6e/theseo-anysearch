"""Faithful CaveDroneSim world export and separately derived fixed-goal tasks.

The upstream exploration policy and lidar map are not used as path-task labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage

from theseo_anysearch.environments.routing_manifests import (
    ConversionRecord,
    GridFrame,
    RightsRecord,
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
from theseo_anysearch.worlds.manifest import WorldExtent

UPSTREAM_URL = "https://github.com/makarov-mm/cave-drone"
UPSTREAM_COMMIT = "ef7852198249390806d8c0cd42e576e02c73c19f"
GENERATOR_FAMILY = "cavedronesim_native_chamber_tunnel_v1"
BRIDGE = Path(__file__).with_name("cave_drone_bridge.cpp")
SOURCE_PATHS = (
    "LICENSE",
    "src/Config.h",
    "src/Math.h",
    "src/Noise.h",
    "src/VoxelDda.h",
    "src/World.cpp",
    "src/World.h",
)
EXPECTED_EXTENT = (192, 56, 192)
EXPECTED_VOXEL_SIZE_M = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream(source_root: Path) -> SourceRecord:
    """Fail closed on revision, local patches, source files and license text."""

    root = source_root.resolve(strict=True)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"CaveDroneSim revision differs from pinned {UPSTREAM_COMMIT}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout
    if dirty:
        raise ValueError("CaveDroneSim tracked source has local modifications")
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("MIT License\n") or "Copyright (c) 2026 Mykhailo Makarov" not in license_text:
        raise ValueError("pinned CaveDroneSim MIT notice is missing or changed")
    files = tuple(
        SourceFile(
            relative_path=relative,
            sha256=_sha256(root / relative),
            role="geometry" if relative != "LICENSE" else "dependency",
        )
        for relative in SOURCE_PATHS
    )
    source = SourceRecord(
        source_id="cavedronesim",
        source_url=UPSTREAM_URL,
        revision=revision,
        files=files,
        rights=RightsRecord(
            status="reviewed",
            license_expression="MIT",
            allowed_uses=("evaluation", "training"),
            evidence=f"LICENSE at {UPSTREAM_COMMIT}; source files verified by SHA-256",
        ),
    )
    for artifact in source.files:
        verify_artifact(root, artifact)
    return source


def compile_bridge(source_root: Path, executable: Path, *, compiler: str = "g++") -> None:
    """Build the tiny exporter against upstream World.cpp, not a Python replica."""

    subprocess.run(
        [
            compiler, "-std=c++23", "-O2", "-I", str(source_root / "src"),
            str(BRIDGE), str(source_root / "src" / "World.cpp"), "-o", str(executable),
        ],
        capture_output=True, text=True, check=True,
    )


def parse_bridge_output(payload: bytes, *, seed: int) -> np.ndarray:
    header, marker, body = payload.partition(b"\n")
    if not marker:
        raise ValueError("CaveDroneSim bridge omitted its header")
    try:
        parts = header.decode("ascii").split(" ")
        if len(parts) != 7 or parts[:2] != ["CAVEDRONE", "1"]:
            raise ValueError("unexpected CaveDroneSim bridge header")
        extent = tuple(int(value) for value in parts[2:5])
        voxel_size = float(parts[5])
        emitted_seed = int(parts[6])
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid CaveDroneSim bridge header") from exc
    if extent != EXPECTED_EXTENT or voxel_size != EXPECTED_VOXEL_SIZE_M:
        raise ValueError("CaveDroneSim grid configuration changed")
    if emitted_seed != seed:
        raise ValueError("CaveDroneSim bridge returned a different seed")
    if len(body) != math.prod(extent):
        raise ValueError("CaveDroneSim bridge returned a truncated or oversized grid")
    array = np.frombuffer(body, dtype=np.uint8).reshape(extent)
    if np.any(array > 1):
        raise ValueError("CaveDroneSim bridge returned non-binary occupancy")
    return array.copy()


def clearance_mask(occupied: np.ndarray, *, body_radius_m: float) -> np.ndarray:
    """Conservative sphere-center clearance from occupied voxel cubes and bounds."""

    if occupied.shape != EXPECTED_EXTENT or occupied.dtype != np.uint8:
        raise ValueError("expected a complete CaveDroneSim uint8 occupancy grid")
    if not math.isfinite(body_radius_m) or body_radius_m < 0:
        raise ValueError("body radius must be a finite nonnegative number")
    free_with_solid_boundary = np.pad(occupied == 0, 1, constant_values=False)
    center_distance_voxels = ndimage.distance_transform_edt(free_with_solid_boundary)[
        1:-1, 1:-1, 1:-1
    ]
    # Lower bound on distance to any occupied voxel cube, not merely its center.
    # Another half voxel covers every point on a 6-neighbor center-to-center edge.
    clearance_m = (
        center_distance_voxels - math.sqrt(3) / 2 - 0.5
    ) * EXPECTED_VOXEL_SIZE_M
    return (occupied == 0) & (clearance_m > body_radius_m)


def select_fixed_goal_pairs(
    occupied: np.ndarray, *, body_radius_m: float
) -> tuple[list[tuple[str, tuple[int, int, int], tuple[int, int, int]]], list[str]]:
    """Select deterministic connected endpoints; record unsatisfied task strata."""

    passable = clearance_mask(occupied, body_radius_m=body_radius_m)
    start = tuple(value // 2 for value in EXPECTED_EXTENT)
    if not passable[start]:
        return [], ["source_start_fails_clearance"]
    six_connected = ndimage.generate_binary_structure(3, 1)
    components, _ = ndimage.label(passable, structure=six_connected)
    members = np.argwhere(components == components[start])
    distances = np.linalg.norm(members - np.asarray(start), axis=1)
    eligible = members[distances >= 16]
    if not len(eligible):
        return [], ["no_connected_goal_at_least_8_m_from_start"]
    farthest = tuple(int(value) for value in eligible[np.argmax(
        np.linalg.norm(eligible - np.asarray(start), axis=1)
    )])
    pairs = [("long_range", start, farthest)]
    altitude_delta = np.abs(eligible[:, 1] - start[1])
    vertical_candidates = eligible[altitude_delta >= 4]
    if len(vertical_candidates):
        vertical_delta = np.abs(vertical_candidates[:, 1] - start[1])
        most_vertical = vertical_candidates[vertical_delta == vertical_delta.max()]
        vertical = tuple(int(value) for value in most_vertical[np.argmax(
            np.linalg.norm(most_vertical - np.asarray(start), axis=1)
        )])
        if vertical != farthest:
            pairs.append(("altitude_change", start, vertical))
        else:
            return pairs, ["altitude_goal_duplicates_long_range_goal"]
    else:
        return pairs, ["no_connected_goal_with_2_m_altitude_change"]
    return pairs, []


def topology_census(occupied: np.ndarray, *, body_radius_m: float) -> dict[str, int]:
    """Count six-connected cavities before and after finite-radius clearance."""

    structure = ndimage.generate_binary_structure(3, 1)
    free = occupied == 0
    passable = clearance_mask(occupied, body_radius_m=body_radius_m)
    _, free_components = ndimage.label(free, structure=structure)
    _, passable_components = ndimage.label(passable, structure=structure)
    start = tuple(value // 2 for value in EXPECTED_EXTENT)
    labels, _ = ndimage.label(passable, structure=structure)
    start_component = labels[start]
    return {
        "free_voxels": int(np.count_nonzero(free)),
        "free_components_6": int(free_components),
        "clearance_passable_voxels": int(np.count_nonzero(passable)),
        "clearance_components_6": int(passable_components),
        "start_clearance_component_voxels": int(np.count_nonzero(
            labels == start_component
        )) if start_component else 0,
    }


def export_seed(
    source_root: Path,
    output_dir: Path,
    *,
    seed: int,
    partition: str,
    body_radius_m: float = 0.25,
    compiler: str = "g++",
) -> dict:
    """Export one immutable seed without treating full truth as a sensor observation."""

    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be a uint32 integer")
    if partition not in {"train", "calibration", "test"}:
        raise ValueError("partition must be train, calibration or test")
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    source = verify_upstream(source_root)
    source.rights.require_allowed("evaluation")
    with tempfile.TemporaryDirectory(prefix="cavedrone-build-") as temporary:
        executable = Path(temporary) / "cave_drone_bridge.exe"
        compile_bridge(source_root, executable, compiler=compiler)
        result = subprocess.run(
            [str(executable), str(seed)], capture_output=True, check=True
        )
    occupied = parse_bridge_output(result.stdout, seed=seed)
    pairs, rejections = select_fixed_goal_pairs(
        occupied, body_radius_m=body_radius_m
    )
    output_dir.mkdir(parents=True)
    occupancy_path = output_dir / "occupancy.npy"
    np.save(occupancy_path, occupied, allow_pickle=False)
    occupancy_sha = _sha256(occupancy_path)
    conversion = ConversionRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        converter_name="cavedronesim-native-bridge",
        converter_version="1",
        parameters={"seed": seed, "bridge_sha256": _sha256(BRIDGE)},
        output_occupancy_sha256=occupancy_sha,
    )
    world = RoutingWorldRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        conversion_identity_sha256=conversion.identity_sha256,
        occupancy_sha256=occupancy_sha,
        extent=WorldExtent(x=EXPECTED_EXTENT[0], y=EXPECTED_EXTENT[1], z=EXPECTED_EXTENT[2]),
        frame=GridFrame(
            source_origin_m=(0.0, 0.0, 0.0),
            meters_per_voxel=EXPECTED_VOXEL_SIZE_M,
        ),
        root_geometry_id=f"cavedronesim-{UPSTREAM_COMMIT[:12]}-seed-{seed}",
        topology_family=GENERATOR_FAMILY,
    )
    tasks = [
        RoutingTaskRecord(
            world_identity_sha256=world.identity_sha256,
            provenance="derived",
            family="drone_flight",
            start_storage=start,
            goal_storage=goal,
            movement_model="6-connected voxel-center existence; swept path not certified",
            body_radius_m=body_radius_m,
            derivation_reason=f"CaveDroneSim full-truth fixed-goal {name} pair; not native exploration",
        )
        for name, start, goal in pairs
    ]
    for task in tasks:
        validate_task_endpoints(task, world, occupied)
    split = RoutingSplitRecord(
        dataset_id=f"cavedronesim-{UPSTREAM_COMMIT[:12]}-seed-{seed}",
        members=(SplitMember(
            world_identity_sha256=world.identity_sha256,
            root_geometry_id=world.root_geometry_id,
            topology_family=world.topology_family,
            partition=partition,
        ),),
    )
    dataset_sha = validate_routing_bundle(
        sources=[source], conversions=[conversion], worlds=[world],
        tasks=tasks, observations=[], references=[], split=split,
    )
    write_sidecar(output_dir / "source.json", source)
    write_sidecar(output_dir / "conversion.json", conversion)
    write_sidecar(output_dir / "world.json", world)
    write_sidecar(output_dir / "split.json", split)
    for index, task in enumerate(tasks):
        write_sidecar(output_dir / f"task-{index:02d}.json", task)
    report = {
        "schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "governing_spec_sha": "1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b",
        "seed": seed,
        "partition": partition,
        "source_identity_sha256": source.content_identity_sha256,
        "world_identity_sha256": world.identity_sha256,
        "dataset_identity_sha256": dataset_sha,
        "occupancy_sha256": occupancy_sha,
        "occupied_voxels": int(np.count_nonzero(occupied)),
        "topology_census": topology_census(occupied, body_radius_m=body_radius_m),
        "task_ids": [task.identity_sha256 for task in tasks],
        "rejected_task_strata": rejections,
        "topology_family": GENERATOR_FAMILY,
        "cross_family_holdout_supported": False,
        "observation_status": "not_exported; full truth is not sensor-visible input",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partition", choices=("train", "calibration", "test"), required=True)
    parser.add_argument("--body-radius-m", type=float, default=0.25)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    report = export_seed(
        args.source, args.output, seed=args.seed, partition=args.partition,
        body_radius_m=args.body_radius_m, compiler=args.compiler,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
