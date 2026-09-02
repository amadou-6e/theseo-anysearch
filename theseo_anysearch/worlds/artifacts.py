"""Unified logical artifact contract for eager and compiled voxel worlds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from theseo_anysearch.cache_lock import cache_key_lock
from theseo_anysearch.worlds.compiler import CompiledWorld, validate_compiled_world
from theseo_anysearch.worlds.manifest import WorldExtent

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_FILE = "geometry-artifact.json"
EAGER_OCCUPANCY_FILE = "occupancy.json"
ARTIFACT_COMPLETE_FILE = "ARTIFACT_COMPLETE"


class GeometryArtifactError(RuntimeError):
    """An artifact is incomplete, corrupt, or incompatible."""


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GeometryArtifactManifest(BaseModel):
    """Representation-independent scene/task metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = ARTIFACT_SCHEMA_VERSION
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extent: WorldExtent
    coordinate_type: Literal["u32"] = "u32"
    storage_coordinate_convention: Literal["zero_based"] = "zero_based"
    environment_coordinate_convention: Literal["one_based"] = "one_based"
    occupancy: Literal["eager_json", "compiled_pack"]
    occupancy_reference: ArtifactReference
    provenance: dict[str, Any] = Field(default_factory=dict)
    transformations: tuple[dict[str, Any], ...] = ()
    candidates: ArtifactReference | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    difficulty: dict[str, Any] = Field(default_factory=dict)
    overview: ArtifactReference | None = None
    algorithms: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "GeometryArtifactManifest":
        payload = self.model_dump(mode="json", exclude={"identity_sha256"})
        if _identity(payload) != self.identity_sha256:
            raise ValueError("geometry artifact identity does not match its metadata")
        return self


class GeometryArtifact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    root: Path
    manifest: GeometryArtifactManifest
    compiled_world: CompiledWorld | None = Field(default=None, exclude=True)

    def eager_coordinates(self) -> tuple[tuple[int, int, int], ...]:
        if self.manifest.occupancy != "eager_json":
            raise GeometryArtifactError("compiled artifacts require regional world access")
        return tuple(tuple(item) for item in json.loads(_checked_read(self.root, self.manifest.occupancy_reference)))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(payload: dict[str, Any]) -> str:
    return _sha(_canonical(payload))


def _checked_read(root: Path, reference: ArtifactReference) -> bytes:
    path = root.joinpath(reference.relative_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GeometryArtifactError(f"artifact payload is unavailable: {path}") from exc
    if _sha(payload) != reference.sha256:
        raise GeometryArtifactError(f"artifact payload checksum mismatch: {path}")
    return payload


def publish_eager_geometry(
    coordinates: tuple[tuple[int, int, int], ...], extent: WorldExtent, cache_root: Path,
    *, provenance: dict[str, Any] | None = None,
    transformations: tuple[dict[str, Any], ...] = (), validation: dict[str, Any] | None = None,
    difficulty: dict[str, Any] | None = None, algorithms: dict[str, str] | None = None,
) -> GeometryArtifact:
    """Atomically publish canonical eager occupancy into a content-addressed cache."""
    occupancy = _canonical(sorted(set(coordinates)))
    reference = ArtifactReference(relative_path=EAGER_OCCUPANCY_FILE, sha256=_sha(occupancy))
    fields = {
        "schema_version": ARTIFACT_SCHEMA_VERSION, "extent": extent.model_dump(mode="json"),
        "coordinate_type": "u32", "storage_coordinate_convention": "zero_based",
        "environment_coordinate_convention": "one_based", "occupancy": "eager_json",
        "occupancy_reference": reference.model_dump(mode="json"), "provenance": provenance or {},
        "transformations": transformations, "candidates": None, "validation": validation or {},
        "difficulty": difficulty or {}, "overview": None, "algorithms": algorithms or {},
    }
    identity = _identity(fields)
    root = cache_root.joinpath(identity)
    cache_root.mkdir(parents=True, exist_ok=True)
    with cache_key_lock(cache_root, identity, timeout_seconds=300.0):
        if root.is_dir():
            return load_geometry_artifact(root)
        temporary = cache_root.joinpath(f".{identity}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            temporary.joinpath(EAGER_OCCUPANCY_FILE).write_bytes(occupancy)
            manifest = GeometryArtifactManifest(identity_sha256=identity, **fields)
            temporary.joinpath(ARTIFACT_MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            temporary.joinpath(ARTIFACT_COMPLETE_FILE).write_text(identity, encoding="ascii")
            os.replace(temporary, root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return load_geometry_artifact(root)


def compiled_geometry_manifest(world: CompiledWorld) -> GeometryArtifactManifest:
    """Project an existing compiled pack into the shared logical contract."""
    source = world.manifest
    reference = ArtifactReference(relative_path="manifest.json", sha256=_sha(world.root.joinpath("manifest.json").read_bytes()))
    fields = {
        "schema_version": 1, "extent": source.extent.model_dump(mode="json"),
        "coordinate_type": source.coordinate_type,
        "storage_coordinate_convention": source.storage_coordinate_convention,
        "environment_coordinate_convention": source.environment_coordinate_convention,
        "occupancy": "compiled_pack", "occupancy_reference": reference.model_dump(mode="json"),
        "provenance": {"source_sha256": source.source_sha256, "compiler": source.compiler},
        "transformations": (), "candidates": None, "validation": {}, "difficulty": {},
        "overview": None if source.overview is None else {"relative_path": source.overview.relative_path, "sha256": source.overview.sha256},
        "algorithms": {"compiler": str(source.compiler.get("schema_version", "unknown"))},
    }
    return GeometryArtifactManifest(identity_sha256=_identity(fields), **fields)


def load_geometry_artifact(root: Path) -> GeometryArtifact:
    """Validate and load eager artifacts or existing compiled packs."""
    artifact_path = root.joinpath(ARTIFACT_MANIFEST_FILE)
    if not artifact_path.is_file():
        compiled = validate_compiled_world(root)
        return GeometryArtifact(root=root, manifest=compiled_geometry_manifest(compiled), compiled_world=compiled)
    try:
        manifest = GeometryArtifactManifest.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GeometryArtifactError("geometry artifact manifest is invalid") from exc
    marker = root.joinpath(ARTIFACT_COMPLETE_FILE)
    if not marker.is_file() or marker.read_text(encoding="ascii") != manifest.identity_sha256:
        raise GeometryArtifactError("geometry artifact is incomplete")
    _checked_read(root, manifest.occupancy_reference)
    return GeometryArtifact(root=root, manifest=manifest)


def migrate_compiled_world(root: Path) -> GeometryArtifactManifest:
    """Validate and project a legacy compiled pack without rewriting it."""
    return compiled_geometry_manifest(validate_compiled_world(root))
