"""Policy action-space settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class ActionConfig(BaseModel):
    """Policy action-space representation."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["discrete_6", "discrete_18", "discrete_26", "vector_3"] = Field("discrete_26", description="Movement action space exposed to the policy.")
