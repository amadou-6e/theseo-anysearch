from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RewardConfig(BaseModel):
    """Base config for a single reward shaper."""
    model_config = ConfigDict(extra="forbid")

    weight: float = 1.0


class CompositeRewardConfig(BaseModel):
    """Config for a pipeline of reward shapers applied in sequence."""
    model_config = ConfigDict(extra="forbid")

    shapers: list[RewardConfig] = []
