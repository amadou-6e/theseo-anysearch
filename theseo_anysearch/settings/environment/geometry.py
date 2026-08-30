"""Geometry source and voxelization settings."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeometryConfig(BaseModel):
    """Geometry source and voxelization settings."""

    model_config = ConfigDict(extra="forbid")
    stl_path: Path | None = Field(None, description="Single STL geometry source.")
    stl_paths: list[Path] | None = Field(None, description="STL sources sampled by the environment.")
    scale: float = Field(1.0, description="Scale applied while voxelizing an STL.")
    scale_range: list[float] | None = Field(None, description="Optional minimum and maximum sampled STL scales.")
    grid_size: int | None = Field(
        default=32,
        ge=1,
        description="Legacy cubic shorthand; omit when an explicit extent is used.",
    )
    extent: tuple[int, int, int] | None = Field(
        default=None,
        description="Independent positive voxel counts for x, y, and z.",
    )
    boxes: list[list[int]] | None = Field(None, description="Inclusive bounds of axis-aligned geometry boxes.")
    pool_size: int = Field(default=0, ge=0, description="Number of generated geometry variants retained in the pool.")
    scale_variants_per_map: int = Field(default=4, ge=1, description="Scale variants generated for each source geometry.")
    padding: int = Field(default=2, ge=0, description="Empty voxel padding around imported geometry.")
    pool: dict[str, Any] | None = Field(None, description="Geometry-pool generation and augmentation settings.")

    @model_validator(mode="before")
    @classmethod
    def resolve_extent_shorthand(cls, value: Any) -> Any:
        """Reject ambiguous bounds while preserving legacy cubic settings."""

        if not isinstance(value, dict) or value.get("extent") is None:
            return value
        data = dict(value)
        extent = tuple(data["extent"])
        if len(extent) != 3 or any(int(axis) < 1 for axis in extent):
            raise ValueError("geometry.extent must contain three positive axes")
        if "grid_size" in data and data["grid_size"] is not None:
            size = int(data["grid_size"])
            if extent != (size, size, size):
                raise ValueError(
                    "geometry.grid_size and extent describe different bounds"
                )
        else:
            data["grid_size"] = None
        return data
