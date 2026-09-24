"""Typed, immutable experiment reference to one verified world and task."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorldSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # "calibration" is a deprecated alias for "validation", accepted only so an
    # existing experiment YAML naming it still parses (see #493).
    role: Literal["train", "validation", "test", "calibration"]
    study_id: str = Field(min_length=1)
    study_root: Path = Path(".")
    root: Path
    world_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_geometry_id: str = Field(min_length=1)
    dataset_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
