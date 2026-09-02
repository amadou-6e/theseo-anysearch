"""Deterministic immutable sparse transformations for compiled worlds."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from theseo_anysearch.worlds.artifacts import GeometryArtifactManifest
from theseo_anysearch.worlds.manifest import WorldExtent

TRANSFORMATION_ALGORITHM_VERSION = "sparse-box-v1"
Coordinate = tuple[int, int, int]


class SparseBox(BaseModel):
    """Inclusive one-based bounds for one immutable overlay obstacle."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum: Coordinate
    maximum_inclusive: Coordinate

    @model_validator(mode="after")
    def ordered(self) -> "SparseBox":
        if any(self.minimum[axis] > self.maximum_inclusive[axis] for axis in range(3)):
            raise ValueError("sparse box minimum exceeds maximum")
        return self

    def contains(self, coordinate: Coordinate) -> bool:
        return all(self.minimum[a] <= coordinate[a] <= self.maximum_inclusive[a] for a in range(3))

    def coordinates_in_region(self, minimum: Coordinate, maximum_exclusive: Coordinate) -> Iterable[Coordinate]:
        starts = tuple(max(self.minimum[a], minimum[a]) for a in range(3))
        stops = tuple(min(self.maximum_inclusive[a] + 1, maximum_exclusive[a]) for a in range(3))
        if any(starts[a] >= stops[a] for a in range(3)):
            return
        for x in range(starts[0], stops[0]):
            for y in range(starts[1], stops[1]):
                for z in range(starts[2], stops[2]):
                    yield x, y, z


class SparseBoxTransform(BaseModel):
    """Content-addressed deterministic augmentation layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    algorithm_version: str = TRANSFORMATION_ALGORITHM_VERSION
    base_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    extent: WorldExtent
    boxes: tuple[SparseBox, ...]
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_identity_and_bounds(self) -> "SparseBoxTransform":
        limits = self.extent.as_tuple()
        for box in self.boxes:
            if any(box.minimum[a] < 1 or box.maximum_inclusive[a] > limits[a] for a in range(3)):
                raise ValueError("sparse box exceeds world extent")
        if self.identity_sha256 != transformation_identity(
            self.base_identity_sha256, self.seed, self.extent, self.boxes
        ):
            raise ValueError("sparse transformation identity mismatch")
        return self


def transformation_identity(base_identity: str, seed: int, extent: WorldExtent, boxes: tuple[SparseBox, ...]) -> str:
    payload = {
        "algorithm_version": TRANSFORMATION_ALGORITHM_VERSION,
        "base_identity_sha256": base_identity, "seed": seed,
        "extent": extent.model_dump(mode="json"),
        "boxes": [box.model_dump(mode="json") for box in boxes],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_box_transform(
    base_identity_sha256: str, extent: WorldExtent, *, seed: int, count: int,
    minimum_size: Coordinate = (1, 1, 1), maximum_size: Coordinate = (4, 4, 4),
) -> SparseBoxTransform:
    """Generate boxes without reading or materializing base occupancy."""
    if count < 0:
        raise ValueError("box count must be non-negative")
    rng = random.Random(seed)
    limits = extent.as_tuple()
    boxes: list[SparseBox] = []
    for _ in range(count):
        size = tuple(rng.randint(minimum_size[a], min(maximum_size[a], limits[a])) for a in range(3))
        start = tuple(rng.randint(1, limits[a] - size[a] + 1) for a in range(3))
        boxes.append(SparseBox(minimum=start, maximum_inclusive=tuple(start[a] + size[a] - 1 for a in range(3))))
    result = tuple(boxes)
    return SparseBoxTransform(
        base_identity_sha256=base_identity_sha256, seed=seed, extent=extent, boxes=result,
        identity_sha256=transformation_identity(base_identity_sha256, seed, extent, result),
    )


class SparseTransformedRead:
    """Overlay-first bounded reader; never enumerates the complete base world."""

    def __init__(self, base_point: Callable[[Coordinate], bool], base_region: Callable[[Coordinate, Coordinate], Iterable[Coordinate]], transform: SparseBoxTransform) -> None:
        self._base_point = base_point
        self._base_region = base_region
        self.transform = transform
        self.point_queries = 0
        self.region_queries = 0

    def occupied(self, coordinate: Coordinate) -> bool:
        self.point_queries += 1
        return any(box.contains(coordinate) for box in self.transform.boxes) or self._base_point(coordinate)

    def occupied_in_region(self, minimum: Coordinate, maximum_exclusive: Coordinate) -> tuple[Coordinate, ...]:
        self.region_queries += 1
        occupied = set(self._base_region(minimum, maximum_exclusive))
        for box in self.transform.boxes:
            occupied.update(box.coordinates_in_region(minimum, maximum_exclusive))
        return tuple(sorted(occupied))


def transformed_artifact_metadata(base: GeometryArtifactManifest, transform: SparseBoxTransform) -> dict[str, Any]:
    """Derive metadata while explicitly invalidating all base occupancy derivatives."""
    if base.identity_sha256 != transform.base_identity_sha256:
        raise ValueError("transformation base identity does not match artifact")
    return {
        "base_artifact_identity": base.identity_sha256,
        "transformed_identity": transform.identity_sha256,
        "transformations": [transform.model_dump(mode="json")],
        "candidates": None,
        "validation": {},
        "difficulty": {},
        "overview": None,
        "derivatives_invalidated": True,
    }


def measure_region_residency(
    reader: SparseTransformedRead,
    minimum: Coordinate,
    maximum_exclusive: Coordinate,
) -> dict[str, float | int]:
    """Measure an identical cold/hot bounded read without assuming a cache backend."""
    started = time.perf_counter()
    cold = reader.occupied_in_region(minimum, maximum_exclusive)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    hot = reader.occupied_in_region(minimum, maximum_exclusive)
    hot_seconds = time.perf_counter() - started
    if cold != hot:
        raise RuntimeError("transformed region changed between cold and hot reads")
    return {
        "cold_seconds": cold_seconds,
        "hot_seconds": hot_seconds,
        "occupied_voxels": len(cold),
    }
