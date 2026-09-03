"""Deterministic procedural observations for perception-encoder pilots."""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from theseo_anysearch.garden.splits import GeometryDescriptor


WORLD_SIDE = 65
_DENSITY_TARGETS = {"low": 0.04, "medium": 0.12, "high": 0.24}
V1_PROGRAM = "voxel-encoder-pilot-v1"
V2_PROGRAM = "voxel-encoder-pilot-v2"
V2R1_PROGRAM = "voxel-encoder-pilot-v2r1"
GENERATOR_VERSION = "procedural-voxel-generator-v2"


def _seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _add_cuboid(
    volume: np.ndarray,
    rng: np.random.Generator,
    *,
    thin: bool,
) -> None:
    side = volume.shape[0]
    if thin:
        sizes = rng.integers(5, 18, size=3)
        sizes[int(rng.integers(0, 3))] = 1
    else:
        sizes = rng.integers(3, 11, size=3)
    starts = [int(rng.integers(0, side - size + 1)) for size in sizes]
    slices = tuple(slice(start, start + int(size)) for start, size in zip(starts, sizes))
    volume[slices] = True


def _add_topology_wall(volume: np.ndarray, rng: np.random.Generator) -> None:
    side = volume.shape[0]
    axis = int(rng.integers(0, 3))
    position = int(rng.integers(5, side - 5))
    wall = [slice(None), slice(None), slice(None)]
    wall[axis] = position
    volume[tuple(wall)] = True
    door_axes = [candidate for candidate in range(3) if candidate != axis]
    door = [slice(None), slice(None), slice(None)]
    door[axis] = position
    for door_axis in door_axes:
        center = int(rng.integers(4, side - 4))
        door[door_axis] = slice(center - 2, center + 3)
    volume[tuple(door)] = False


def _add_ellipsoid(volume: np.ndarray, rng: np.random.Generator) -> None:
    side = volume.shape[0]
    center = rng.uniform(4, side - 5, size=3)
    radii = rng.uniform(2.0, 8.0, size=3)
    lower = np.maximum(0, np.floor(center - radii).astype(int))
    upper = np.minimum(side, np.ceil(center + radii + 1).astype(int))
    coordinates = np.ogrid[
        lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]
    ]
    distance = sum(
        ((coordinate - center[index]) / radii[index]) ** 2
        for index, coordinate in enumerate(coordinates)
    )
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    volume[slices] |= distance <= 1


@lru_cache(maxsize=256)
def _canonical_world(
    program: str,
    geometry_id: str,
    family: str,
    occupancy_band: str,
) -> np.ndarray:
    """Create one immutable canonical world for a frozen geometry identity."""

    seed_parts = (
        (V1_PROGRAM, geometry_id)
        if program == V1_PROGRAM
        else (GENERATOR_VERSION, program, geometry_id)
    )
    rng = np.random.default_rng(_seed(*seed_parts))
    volume = np.zeros((WORLD_SIDE,) * 3, dtype=bool)
    target = _DENSITY_TARGETS[occupancy_band]
    attempts = 0
    while float(volume.mean()) < target and attempts < 2_000:
        if family == "open":
            _add_cuboid(volume, rng, thin=False)
        elif family == "thin_obstacle":
            _add_cuboid(volume, rng, thin=True)
        elif family == "topology":
            _add_topology_wall(volume, rng)
        elif family == "imported":
            _add_ellipsoid(volume, rng)
        else:
            raise ValueError(f"unsupported geometry family: {family}")
        attempts += 1
    if not volume.any() or volume.all():
        raise RuntimeError(f"failed to generate nontrivial geometry {geometry_id}")
    volume.setflags(write=False)
    return volume


@dataclass(frozen=True)
class PilotObservation:
    geometry_id: str
    observation_id: str
    family: str
    occupancy_band: str
    radius: int
    occupancy: np.ndarray
    unknown_mask: np.ndarray
    identity_sha256: str


def make_pilot_observation(
    descriptor: GeometryDescriptor,
    observation_index: int,
    *,
    radius: int,
    density_multiplier: int = 1,
    program: str = V1_PROGRAM,
) -> PilotObservation:
    """Materialize a reproducible crop without mutating the cached source world."""

    if observation_index < 0 or radius not in {8, 16, 32}:
        raise ValueError("observation index must be nonnegative and radius supported")
    if density_multiplier not in {1, 4}:
        raise ValueError("density multiplier must be one or four")
    if program not in {V1_PROGRAM, V2_PROGRAM, V2R1_PROGRAM}:
        raise ValueError(f"unsupported pilot corpus program: {program}")
    world = _canonical_world(
        program, descriptor.geometry_id, descriptor.family, descriptor.occupancy_band
    )
    side = 2 * radius + 1
    maximum_start = WORLD_SIDE - side
    observation_seed = (
        (
            "observation",
            descriptor.geometry_id,
            observation_index,
            radius,
            density_multiplier,
        )
        if program == V1_PROGRAM
        else (
            GENERATOR_VERSION,
            program,
            "observation",
            descriptor.geometry_id,
            observation_index,
            radius,
            density_multiplier,
        )
    )
    rng = np.random.default_rng(_seed(*observation_seed))
    starts = (0, 0, 0)
    occupancy = np.zeros((side,) * 3, dtype=bool)
    for _ in range(64):
        starts = tuple(int(rng.integers(0, maximum_start + 1)) for _ in range(3))
        slices = tuple(slice(start, start + side) for start in starts)
        occupancy = np.array(world[slices], copy=True)
        if occupancy.any():
            break
    if not occupancy.any():
        occupied_cells = np.argwhere(world)
        anchor = occupied_cells[
            _seed(descriptor.geometry_id, observation_index, radius) % len(occupied_cells)
        ]
        starts = tuple(
            int(np.clip(coordinate - radius, 0, maximum_start))
            for coordinate in anchor
        )
    slices = tuple(slice(start, start + side) for start in starts)
    occupancy = np.array(world[slices], copy=True)
    unknown = np.zeros_like(occupancy)
    if observation_index % 5 == 0:
        axis = observation_index % 3
        pattern = (observation_index // 5) % 2
        width = max(1, side // 5)
        hidden = [slice(None), slice(None), slice(None)]
        if pattern == 0:
            hidden[axis] = slice(0, width)
        else:
            middle = side // 2
            hidden[axis] = slice(middle - width // 2, middle - width // 2 + width)
        unknown[tuple(hidden)] = True
        occupancy[unknown] = False
    if not occupancy.any():
        original = np.asarray(world[slices])
        coordinate = tuple(int(value) for value in np.argwhere(original)[0])
        unknown[coordinate] = False
        occupancy[coordinate] = True
    observation_id = (
        f"{descriptor.geometry_id}:d{density_multiplier}:r{radius}:o{observation_index:05d}"
    )
    digest = hashlib.sha256()
    digest.update(observation_id.encode("ascii"))
    digest.update(occupancy.tobytes(order="C"))
    digest.update(unknown.tobytes(order="C"))
    return PilotObservation(
        geometry_id=descriptor.geometry_id,
        observation_id=observation_id,
        family=descriptor.family,
        occupancy_band=descriptor.occupancy_band,
        radius=radius,
        occupancy=occupancy,
        unknown_mask=unknown,
        identity_sha256=digest.hexdigest(),
    )


def proper_cube_rotation(volume: np.ndarray, rotation_index: int) -> np.ndarray:
    """Apply one of the 24 proper cube rotations in a stable enumeration."""

    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("cube rotation expects one cubic volume")
    rotations: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        permutation_sign = -1 if inversions % 2 else 1
        for signs in itertools.product((-1, 1), repeat=3):
            if permutation_sign * signs[0] * signs[1] * signs[2] != 1:
                continue
            candidate = np.transpose(volume, permutation)
            for axis, sign in enumerate(signs):
                if sign < 0:
                    candidate = np.flip(candidate, axis=axis)
            rotations.append(candidate.copy())
    if len(rotations) != 24:
        raise AssertionError("proper rotation enumeration must contain 24 elements")
    return rotations[rotation_index % 24]
