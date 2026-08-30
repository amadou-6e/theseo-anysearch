"""Coordinate, extent, and identity contracts for finite voxel worlds."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WORLD_SCHEMA_VERSION = 1
COORDINATE_TYPE = "u32"
STORAGE_COORDINATE_CONVENTION = "zero_based"
ENVIRONMENT_COORDINATE_CONVENTION = "one_based"


class WorldExtent(BaseModel):
    """Positive voxel counts for the three independent world axes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int = Field(ge=1, le=2**32 - 1)
    y: int = Field(ge=1, le=2**32 - 1)
    z: int = Field(ge=1, le=2**32 - 1)

    @classmethod
    def from_value(cls, value: int | tuple[int, int, int] | list[int]) -> WorldExtent:
        """Resolve cubic shorthand or an explicit three-axis extent."""

        if isinstance(value, bool):
            raise TypeError("world extent must be an integer or three-axis sequence")
        if isinstance(value, int):
            return cls(x=value, y=value, z=value)
        if not isinstance(value, (tuple, list)):
            raise TypeError("world extent must be an integer or three-axis sequence")
        if len(value) != 3:
            raise ValueError("world extent must contain exactly three axes")
        return cls(x=value[0], y=value[1], z=value[2])

    def as_tuple(self) -> tuple[int, int, int]:
        """Return the canonical axis order."""

        return self.x, self.y, self.z


class ChunkCoordinate(BaseModel):
    """Zero-based chunk-space coordinate used as a tuple global key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int = Field(ge=0, le=2**32 - 1)
    y: int = Field(ge=0, le=2**32 - 1)
    z: int = Field(ge=0, le=2**32 - 1)


class WorldChunkManifest(BaseModel):
    """Identity and integrity metadata for one immutable world chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coordinate: ChunkCoordinate
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)
    occupied_voxels: int = Field(ge=0)
    encoding: Literal["uniform", "sparse_u32", "dense_zlib"] = "sparse_u32"
    pack_offset: int = Field(default=0, ge=0)
    decoded_byte_length: int = Field(default=0, ge=0)


class WorldManifest(BaseModel):
    """Foundational identity for a compiled large finite world."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = WORLD_SCHEMA_VERSION
    coordinate_type: Literal["u32"] = COORDINATE_TYPE
    storage_coordinate_convention: Literal["zero_based"] = (
        STORAGE_COORDINATE_CONVENTION
    )
    environment_coordinate_convention: Literal["one_based"] = (
        ENVIRONMENT_COORDINATE_CONVENTION
    )
    environment_min: tuple[int, int, int] = (1, 1, 1)
    source_origin: tuple[int, int, int] = (0, 0, 0)
    extent: WorldExtent
    chunk_shape: WorldExtent
    chunks: tuple[WorldChunkManifest, ...]
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    voxel_scale: float = Field(default=1.0, gt=0.0)
    compiler: dict[str, Any] = Field(default_factory=dict)
    pack_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_environment_min(self) -> WorldManifest:
        """Keep the existing public one-based task coordinate contract."""

        if self.environment_min != (1, 1, 1):
            raise ValueError("one-based worlds require environment_min [1, 1, 1]")
        coordinates = [
            chunk.coordinate.model_dump(mode="json") for chunk in self.chunks
        ]
        if len({(item["x"], item["y"], item["z"]) for item in coordinates}) != len(
            coordinates
        ):
            raise ValueError("world chunks must have unique tuple coordinates")
        chunk_limits = (
            (self.extent.x - 1) // self.chunk_shape.x,
            (self.extent.y - 1) // self.chunk_shape.y,
            (self.extent.z - 1) // self.chunk_shape.z,
        )
        for chunk in self.chunks:
            if (
                chunk.coordinate.x > chunk_limits[0]
                or chunk.coordinate.y > chunk_limits[1]
                or chunk.coordinate.z > chunk_limits[2]
            ):
                raise ValueError("world chunk coordinate exceeds the finite extent")
        return self


def world_contract(env_config: dict[str, Any]) -> dict[str, Any]:
    """Return canonical world fields used by every compatibility fingerprint."""

    raw_extent = env_config.get("extent")
    scalar_size = env_config.get("grid_size")
    if raw_extent is not None and scalar_size is not None:
        explicit = WorldExtent.from_value(raw_extent)
        shorthand = WorldExtent.from_value(int(scalar_size))
        if explicit != shorthand:
            raise ValueError("grid_size and extent describe different world bounds")
    if raw_extent is None:
        raw_extent = int(scalar_size if scalar_size is not None else 32)
    extent = WorldExtent.from_value(raw_extent)
    raw_origin = env_config.get("source_origin", (0, 0, 0))
    if not isinstance(raw_origin, (tuple, list)) or len(raw_origin) != 3:
        raise ValueError("source_origin must contain exactly three axes")
    return {
        "schema_version": WORLD_SCHEMA_VERSION,
        "coordinate_type": COORDINATE_TYPE,
        "storage_coordinate_convention": STORAGE_COORDINATE_CONVENTION,
        "environment_coordinate_convention": ENVIRONMENT_COORDINATE_CONVENTION,
        "environment_min": [1, 1, 1],
        "source_origin": [int(value) for value in raw_origin],
        "extent": list(extent.as_tuple()),
        "identity_sha256": env_config.get("world_identity_sha256"),
    }


def world_contract_fingerprint(contract: dict[str, Any]) -> str:
    """Hash a canonical contract for checkpoint and artifact compatibility."""

    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
