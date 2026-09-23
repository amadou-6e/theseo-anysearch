"""Portable contracts for imported 3D routing worlds and their tasks."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, model_validator

from theseo_anysearch.worlds.manifest import (
    WorldExtent,
    world_contract_fingerprint as contract_identity,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StorageCoordinate = tuple[StrictInt, StrictInt, StrictInt]
Use = Literal["evaluation", "training", "redistribution"]
# "calibration" is a deprecated alias for "validation", accepted only so an
# already-written, content-addressed split.json naming it keeps validating and
# hashing to its original identity_sha256 (see #493). Never write "calibration".
Partition = Literal["train", "validation", "test", "calibration"]
SCHEMA_VERSION = 1


class RoutingRecord(BaseModel):
    """Base for strict, content-addressed JSON sidecars."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = SCHEMA_VERSION

    def canonical_payload(self) -> dict:
        return self.model_dump(mode="json")

    @property
    def identity_sha256(self) -> str:
        return contract_identity(
            {"record_type": type(self).__name__, "payload": self.canonical_payload()}
        )


class ArtifactRef(RoutingRecord):
    relative_path: Name
    sha256: Sha256

    @model_validator(mode="after")
    def require_portable_path(self) -> ArtifactRef:
        path = self.relative_path
        if (
            "\\" in path
            or ":" in path.split("/", 1)[0]
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("artifact path must be a normalized relative POSIX path")
        return self


def verify_artifact(root: Path, artifact: ArtifactRef) -> None:
    """Verify a local source file without following a path outside its source root."""

    base = root.resolve(strict=True)
    path = base.joinpath(artifact.relative_path).resolve(strict=True)
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("artifact path escapes its source root or is not a file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != artifact.sha256:
        raise ValueError("artifact SHA-256 does not match its source file")


class SourceFile(ArtifactRef):
    role: Literal["geometry", "task", "observation", "reference", "dependency"]


class RightsRecord(RoutingRecord):
    status: Literal["unreviewed", "reviewed"] = "unreviewed"
    license_expression: str | None = None
    allowed_uses: tuple[Use, ...] = ()
    evidence: str | None = None

    @model_validator(mode="after")
    def require_review_evidence(self) -> RightsRecord:
        if len(set(self.allowed_uses)) != len(self.allowed_uses):
            raise ValueError("allowed uses must be unique")
        if self.status == "unreviewed" and self.allowed_uses:
            raise ValueError("unreviewed rights cannot allow any use")
        if self.status == "reviewed" and not self.evidence:
            raise ValueError("reviewed rights require evidence")
        return self

    def canonical_payload(self) -> dict:
        payload = super().canonical_payload()
        payload["allowed_uses"] = sorted(payload["allowed_uses"])
        return payload

    def require_allowed(self, use: Use) -> None:
        if self.status != "reviewed" or use not in self.allowed_uses:
            raise PermissionError(f"{use} is not cleared for this source")


class SourceRecord(RoutingRecord):
    source_id: Name
    source_url: Name
    revision: Name
    files: tuple[SourceFile, ...] = Field(min_length=1)
    rights: RightsRecord = Field(default_factory=RightsRecord)

    @model_validator(mode="after")
    def require_unique_files(self) -> SourceRecord:
        paths = [item.relative_path for item in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("source files must have unique paths")
        return self

    def canonical_payload(self) -> dict:
        payload = super().canonical_payload()
        payload["files"] = sorted(payload["files"], key=lambda item: item["relative_path"])
        payload["rights"] = self.rights.canonical_payload()
        return payload

    @property
    def content_identity_sha256(self) -> str:
        """Rights changes affect the dataset, not the source geometry identity."""

        return contract_identity(
            {
                "record_type": "SourceContent",
                "schema_version": self.schema_version,
                "source_id": self.source_id,
                "source_url": self.source_url,
                "revision": self.revision,
                "files": sorted(
                    (item.model_dump(mode="json") for item in self.files),
                    key=lambda item: item["relative_path"],
                ),
            }
        )


class GridFrame(RoutingRecord):
    """Map zero-based voxel centers to positions in the source metric frame."""

    source_origin_m: tuple[float, float, float]
    storage_axes_in_source: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    meters_per_voxel: float = Field(gt=0)

    @model_validator(mode="after")
    def require_orthonormal_axes(self) -> GridFrame:
        axes = self.storage_axes_in_source
        for axis in axes:
            if not math.isclose(sum(value * value for value in axis), 1.0, abs_tol=1e-6):
                raise ValueError("storage axes must be unit vectors")
        for left in range(3):
            for right in range(left + 1, 3):
                dot = sum(axes[left][index] * axes[right][index] for index in range(3))
                if not math.isclose(dot, 0.0, abs_tol=1e-6):
                    raise ValueError("storage axes must be orthogonal")
        return self

    def to_source_center(
        self, coordinate: Sequence[int], extent: WorldExtent
    ) -> tuple[float, float, float]:
        storage = _storage_coordinate(coordinate, extent)
        return tuple(
            self.source_origin_m[source_axis]
            + self.meters_per_voxel
            * sum(
                (storage[storage_axis] + 0.5)
                * self.storage_axes_in_source[storage_axis][source_axis]
                for storage_axis in range(3)
            )
            for source_axis in range(3)
        )  # type: ignore[return-value]

    def from_source_center(
        self, source_m: Sequence[float], extent: WorldExtent
    ) -> StorageCoordinate:
        if len(source_m) != 3 or not all(math.isfinite(value) for value in source_m):
            raise ValueError("source coordinate must contain three finite meters")
        offset = tuple(source_m[index] - self.source_origin_m[index] for index in range(3))
        values: list[int] = []
        for axis in self.storage_axes_in_source:
            raw = sum(offset[index] * axis[index] for index in range(3))
            raw = raw / self.meters_per_voxel - 0.5
            nearest = round(raw)
            if not math.isclose(raw, nearest, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("source coordinate is not a voxel center")
            values.append(nearest)
        return _storage_coordinate(values, extent)


def _storage_coordinate(
    coordinate: Sequence[int], extent: WorldExtent
) -> StorageCoordinate:
    if len(coordinate) != 3 or any(isinstance(value, bool) for value in coordinate):
        raise ValueError("storage coordinate must contain three integer axes")
    try:
        values = tuple(int(operator.index(value)) for value in coordinate)
    except TypeError as exc:
        raise ValueError("storage coordinate must contain three integer axes") from exc
    if any(not 0 <= values[index] < extent.as_tuple()[index] for index in range(3)):
        raise ValueError("storage coordinate is outside the world extent")
    return values[0], values[1], values[2]


def storage_to_task(
    coordinate: Sequence[int], extent: WorldExtent
) -> StorageCoordinate:
    values = _storage_coordinate(coordinate, extent)
    return values[0] + 1, values[1] + 1, values[2] + 1


def task_to_storage(
    coordinate: Sequence[int], extent: WorldExtent
) -> StorageCoordinate:
    if len(coordinate) != 3 or any(isinstance(value, bool) for value in coordinate):
        raise ValueError("task coordinate must contain three integer axes")
    try:
        storage = tuple(int(operator.index(value)) - 1 for value in coordinate)
    except TypeError as exc:
        raise ValueError("task coordinate must contain three integer axes") from exc
    return _storage_coordinate(storage, extent)


class ConversionRecord(RoutingRecord):
    source_content_identity_sha256: Sha256
    converter_name: Name
    converter_version: Name
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    output_occupancy_sha256: Sha256
    parent_world_identity_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def require_named_parameters(self) -> ConversionRecord:
        if any(not key.strip() for key in self.parameters):
            raise ValueError("conversion parameters must have nonempty names")
        return self


class RoutingWorldRecord(RoutingRecord):
    source_content_identity_sha256: Sha256
    conversion_identity_sha256: Sha256
    occupancy_sha256: Sha256
    truth_coverage: Literal["complete"] = "complete"
    extent: WorldExtent
    frame: GridFrame
    root_geometry_id: Name
    topology_family: Name
    site_id: Name | None = None
    parent_world_identity_sha256: Sha256 | None = None


class RoutingTaskRecord(RoutingRecord):
    world_identity_sha256: Sha256
    provenance: Literal["upstream", "derived"]
    family: Literal["point_path", "drone_flight", "single_pipe", "coupled_pipes"]
    start_storage: StorageCoordinate
    goal_storage: StorageCoordinate
    movement_model: Name
    body_radius_m: float = Field(default=0.0, ge=0)
    ceiling_source_m: float | None = None
    source_query_id: str | None = None
    source_task_sha256: Sha256 | None = None
    derivation_reason: str | None = None

    @model_validator(mode="after")
    def require_provenance(self) -> RoutingTaskRecord:
        if any(value < 0 for value in (*self.start_storage, *self.goal_storage)):
            raise ValueError("task storage coordinates must be nonnegative")
        if self.provenance == "upstream":
            if not self.source_query_id or not self.source_task_sha256 or self.derivation_reason:
                raise ValueError("upstream tasks require source query evidence and no derivation")
        elif not self.derivation_reason:
            raise ValueError("derived tasks require a derivation reason")
        return self


class RoutingObservationRecord(RoutingRecord):
    world_identity_sha256: Sha256
    sensor_model: Name
    observed_occupied: ArtifactRef
    observed_free: ArtifactRef
    unknown: ArtifactRef
    sensor_config_sha256: Sha256 | None = None


class RoutingReferenceRecord(RoutingRecord):
    task_identity_sha256: Sha256
    claim: Literal["unverified", "feasible", "independently_validated", "certified_optimal"]
    route_artifact: ArtifactRef | None = None
    cost: float | None = Field(default=None, ge=0)
    verification_evidence: str | None = None

    @model_validator(mode="after")
    def require_claim_evidence(self) -> RoutingReferenceRecord:
        if self.claim != "unverified" and self.route_artifact is None:
            raise ValueError("a route is required for a reference feasibility claim")
        if (
            self.claim in {"independently_validated", "certified_optimal"}
            and not self.verification_evidence
        ):
            raise ValueError("independent validation requires verification evidence")
        return self


class SplitMember(RoutingRecord):
    world_identity_sha256: Sha256
    root_geometry_id: Name
    topology_family: Name
    site_id: Name | None = None
    partition: Partition


class RoutingSplitRecord(RoutingRecord):
    dataset_id: Name
    members: tuple[SplitMember, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_group_leakage(self) -> RoutingSplitRecord:
        seen_worlds: set[str] = set()
        roots: dict[str, Partition] = {}
        sites: dict[str, Partition] = {}
        for member in self.members:
            if member.world_identity_sha256 in seen_worlds:
                raise ValueError("split contains a duplicate world")
            seen_worlds.add(member.world_identity_sha256)
            for group, key, assignments in (
                ("root geometry", member.root_geometry_id, roots),
                ("site", member.site_id, sites),
            ):
                if key is not None and key in assignments and assignments[key] != member.partition:
                    raise ValueError(f"{group} crosses split partitions")
                if key is not None:
                    assignments[key] = member.partition
        return self

    def canonical_payload(self) -> dict:
        payload = super().canonical_payload()
        payload["members"] = sorted(
            payload["members"], key=lambda item: item["world_identity_sha256"]
        )
        return payload


def validate_observation_masks(
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    *,
    full_occupied: np.ndarray | None = None,
) -> None:
    """Validate a compact observation without interpreting unknown as free."""

    values = [np.asarray(item) for item in (observed_occupied, observed_free, unknown)]
    if full_occupied is not None:
        values.append(np.asarray(full_occupied))
    if any(item.ndim != 3 or item.shape != values[0].shape for item in values):
        raise ValueError("observation masks must have the same 3D shape")
    if any(
        item.dtype.kind not in "biu" or not np.all((item == 0) | (item == 1))
        for item in values
    ):
        raise ValueError("observation masks must contain only Boolean or 0/1 values")
    occupied, free, hidden = (item.astype(bool, copy=False) for item in values[:3])
    if np.any(occupied & free) or np.any(occupied & hidden) or np.any(free & hidden):
        raise ValueError("observed occupied, free and unknown masks must be disjoint")
    if not np.all(occupied | free | hidden):
        raise ValueError("observation masks must cover the full grid")
    if full_occupied is not None:
        truth = values[3].astype(bool, copy=False)
        if np.any(occupied & ~truth) or np.any(free & truth):
            raise ValueError("observed occupancy conflicts with full-world truth")


def validate_task_endpoints(
    task: RoutingTaskRecord,
    world: RoutingWorldRecord,
    full_occupied: np.ndarray,
) -> None:
    """Reject invalid point endpoints before any planner-specific search."""

    if task.world_identity_sha256 != world.identity_sha256:
        raise ValueError("task references a different world")
    occupancy = np.asarray(full_occupied)
    if occupancy.shape != world.extent.as_tuple() or occupancy.ndim != 3:
        raise ValueError("full-world occupancy shape does not match extent")
    if occupancy.dtype.kind not in "biu" or not np.all(
        (occupancy == 0) | (occupancy == 1)
    ):
        raise ValueError("full-world occupancy must contain Boolean or 0/1 values")
    start = _storage_coordinate(task.start_storage, world.extent)
    goal = _storage_coordinate(task.goal_storage, world.extent)
    if occupancy[start]:
        raise ValueError("occupied start")
    if occupancy[goal]:
        raise ValueError("occupied goal")


def validate_routing_bundle(
    *,
    sources: Sequence[SourceRecord],
    conversions: Sequence[ConversionRecord],
    worlds: Sequence[RoutingWorldRecord],
    tasks: Sequence[RoutingTaskRecord],
    observations: Sequence[RoutingObservationRecord],
    references: Sequence[RoutingReferenceRecord],
    split: RoutingSplitRecord,
) -> str:
    """Check sidecar links and return a path-independent dataset identity."""

    source_by_id = {source.content_identity_sha256: source for source in sources}
    conversion_by_id = {item.identity_sha256: item for item in conversions}
    world_by_id = {item.identity_sha256: item for item in worlds}
    task_by_id = {item.identity_sha256: item for item in tasks}
    observation_ids = {item.identity_sha256 for item in observations}
    reference_ids = {item.identity_sha256 for item in references}
    for records, unique in (
        (sources, source_by_id),
        (conversions, conversion_by_id),
        (worlds, world_by_id),
        (tasks, task_by_id),
        (observations, observation_ids),
        (references, reference_ids),
    ):
        if len(records) != len(unique):
            raise ValueError("routing bundle contains duplicate record identities")
    for conversion in conversions:
        if conversion.source_content_identity_sha256 not in source_by_id:
            raise ValueError("conversion references an unknown source")
    for world in worlds:
        conversion = conversion_by_id.get(world.conversion_identity_sha256)
        if conversion is None:
            raise ValueError("world references an unknown conversion")
        if (
            world.source_content_identity_sha256 != conversion.source_content_identity_sha256
            or world.occupancy_sha256 != conversion.output_occupancy_sha256
            or world.parent_world_identity_sha256 != conversion.parent_world_identity_sha256
        ):
            raise ValueError("world and conversion provenance disagree")
        parent = world_by_id.get(world.parent_world_identity_sha256)
        if world.parent_world_identity_sha256 is not None and parent is None:
            raise ValueError("derived world references an unknown parent world")
        if parent is not None:
            if (
                parent.root_geometry_id != world.root_geometry_id
                or parent.source_content_identity_sha256
                != world.source_content_identity_sha256
                or parent.site_id != world.site_id
            ):
                raise ValueError("derived world changed its source lineage")
    for task in tasks:
        world = world_by_id.get(task.world_identity_sha256)
        if world is None:
            raise ValueError("task references an unknown world")
        _storage_coordinate(task.start_storage, world.extent)
        _storage_coordinate(task.goal_storage, world.extent)
        source = source_by_id[world.source_content_identity_sha256]
        if task.source_task_sha256 is not None and task.source_task_sha256 not in {
            item.sha256 for item in source.files if item.role == "task"
        }:
            raise ValueError("task references a source query file not in its source record")
    for observation in observations:
        if observation.world_identity_sha256 not in world_by_id:
            raise ValueError("observation references an unknown world")
    for reference in references:
        if reference.task_identity_sha256 not in task_by_id:
            raise ValueError("reference references an unknown task")
    member_ids = {item.world_identity_sha256 for item in split.members}
    if member_ids != set(world_by_id):
        raise ValueError("split must assign every world exactly once")
    for member in split.members:
        world = world_by_id[member.world_identity_sha256]
        if (
            member.root_geometry_id != world.root_geometry_id
            or member.site_id != world.site_id
            or member.topology_family != world.topology_family
        ):
            raise ValueError("split metadata disagrees with the world record")
    return contract_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "sources": sorted(source.identity_sha256 for source in sources),
            "conversions": sorted(item.identity_sha256 for item in conversions),
            "worlds": sorted(world_by_id),
            "tasks": sorted(task_by_id),
            "observations": sorted(observation_ids),
            "references": sorted(reference_ids),
            "split": split.identity_sha256,
        }
    )


RecordT = TypeVar("RecordT", bound=RoutingRecord)


def write_sidecar(path: Path, record: RoutingRecord) -> None:
    """Write an immutable canonical JSON envelope; never overwrite an existing sidecar."""

    payload = record.canonical_payload()
    envelope = {
        "record_type": type(record).__name__,
        "identity_sha256": record.identity_sha256,
        "payload": payload,
    }
    encoded = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


def read_sidecar(path: Path, record_type: type[RecordT]) -> RecordT:
    """Reject malformed, mistyped or content-tampered sidecars."""

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    envelope = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(envelope, dict) or set(envelope) != {
        "record_type",
        "identity_sha256",
        "payload",
    }:
        raise ValueError("invalid routing sidecar envelope")
    if envelope["record_type"] != record_type.__name__:
        raise ValueError("routing sidecar has the wrong record type")
    record = record_type.model_validate(envelope["payload"])
    if record.identity_sha256 != envelope["identity_sha256"]:
        raise ValueError("routing sidecar identity does not match its content")
    return record
