"""Geometry source and voxelization settings."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeometryValidationConfig(BaseModel):
    """Opt-in structural and task-feasibility validation budgets."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    maximum_attempts: int = Field(default=1, ge=1)
    maximum_search_nodes: int = Field(default=100_000, ge=1)
    recovery_margin_steps: int = Field(default=0, ge=0)
    clearance_radius: int | None = Field(default=None, ge=1)
    difficulty_bands: tuple["RoutingDifficultyBand", ...] = ()
    accepted_difficulty_bands: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_bands(self) -> "GeometryValidationConfig":
        names = [band.name for band in self.difficulty_bands]
        if len(names) != len(set(names)):
            raise ValueError("geometry validation difficulty-band names must be unique")
        unknown = set(self.accepted_difficulty_bands) - set(names)
        if unknown:
            raise ValueError(f"accepted difficulty bands are not defined: {sorted(unknown)}")
        return self


class NumericRange(BaseModel):
    """Inclusive numeric interval used by a routing-difficulty band."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "NumericRange":
        if self.minimum is None and self.maximum is None:
            raise ValueError("difficulty range must define minimum or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("difficulty range minimum cannot exceed maximum")
        return self


class RoutingDifficultyBand(BaseModel):
    """Named conjunction of routing-descriptor ranges."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    path_length: NumericRange | None = None
    detour_ratio: NumericRange | None = None
    direction_changes: NumericRange | None = None
    vertical_displacement: NumericRange | None = None
    expansion_count: NumericRange | None = None

    @model_validator(mode="after")
    def require_constraint(self) -> "RoutingDifficultyBand":
        fields = (
            self.path_length,
            self.detour_ratio,
            self.direction_changes,
            self.vertical_displacement,
            self.expansion_count,
        )
        if not any(item is not None for item in fields):
            raise ValueError("difficulty band must constrain at least one descriptor")
        return self


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
    validation: GeometryValidationConfig = Field(
        default_factory=GeometryValidationConfig,
        description="Shared geometry and navigation-task validation settings.",
    )
    compiled_world_path: Path | None = Field(
        None, description="Validated compiled-world directory loaded lazily by each worker."
    )
    maximum_decoded_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1,
        description="Per-process upper bound for unpinned decoded world chunks.",
    )
    prefetch_margin: int = Field(
        default=2,
        ge=0,
        description="Extra voxels loaded around the observation and movement envelope.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_extent_shorthand(cls, value: Any) -> Any:
        """Reject ambiguous bounds while preserving legacy cubic settings."""

        if isinstance(value, dict):
            pool = value.get("pool")
            feasibility = (
                ((pool or {}).get("augmentation") or {}).get("feasibility")
                if isinstance(pool, dict)
                else None
            )
            if feasibility is not None:
                if not isinstance(feasibility, dict):
                    raise ValueError("geometry.pool.augmentation.feasibility must be an object")
                if feasibility.get("enabled", True):
                    for key in ("maximum_attempts", "maximum_search_nodes"):
                        if int(feasibility.get(key, 0)) < 1:
                            raise ValueError(
                                f"geometry.pool.augmentation.feasibility.{key} must be positive"
                            )
                    if int(feasibility.get("recovery_margin_steps", 0)) < 0:
                        raise ValueError(
                            "geometry.pool.augmentation.feasibility.recovery_margin_steps "
                            "must be non-negative"
                        )
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
