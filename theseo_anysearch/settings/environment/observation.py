"""Policy observation settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class ObservationConfig(BaseModel):
    """Policy observation representation."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["scalar", "box", "radial", "hierarchical_box"] = Field("scalar", description="Observation encoder exposed to the policy.")
    box_radius: int = Field(default=2, ge=0, description="Voxel radius of a local box observation.")
    box_radii: list[int] | None = Field(None, description="Radii combined by a hierarchical box observation.")
    ray_max_len: int = Field(default=16, ge=1, description="Maximum radial-observation ray length in voxels.")
