"""Reward settings and provider selection."""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

class RewardSelector(BaseModel):
    """Select a named built-in, Python, or Rust reward provider."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", description="Registered reward-provider name.")
    parameters: dict[str, JsonValue] = Field(default_factory=dict, description="JSON-compatible parameters passed to the selected provider.")


class RewardConfig(BaseModel):
    """Reward terms computed by the voxel environment."""

    model_config = ConfigDict(extra="forbid")
    step_cost: float = Field(-0.01, description="Reward applied on every environment step.")
    collision_cost: float = Field(0.0, description="Additional reward applied when movement collides.")
    goal_reward: float = Field(1.0, description="Reward added when the goal state is reached.")
    distance_shaping: float = Field(0.0, description="Coefficient applied to normalized goal progress.")
    distance_reward_mode: Literal["progress", "zone"] = Field("progress", description="Distance-based reward calculation mode.")
    zone_reward_min: float = Field(-1.0, description="Most negative reward emitted by zone mode.")
    zone_reward_max: float = Field(-0.01, description="Least negative non-terminal reward emitted by zone mode.")
    zone_reward_curve: Literal["linear", "exponential"] = Field("linear", description="Interpolation curve used by zone mode.")
    distance_metric: Literal["euclidean", "manhattan"] = Field("euclidean", description="Distance metric used by reward calculations.")
    invalid_action_cost: float = Field(0.0, description="Additional reward applied to invalid action encodings.")
    construction_residual_weight: float = Field(default=0.0, ge=0.0, description="Penalty weight for construction residual or overshoot.")
    construction_overshoot_weight: float = Field(default=0.0, ge=0.0, description="Penalty weight for construction residual or overshoot.")
    provider: RewardSelector | None = Field(default=None, validation_alias=AliasChoices("provider", "custom"), description="Named reward implementation; the legacy YAML key custom remains accepted.")

    @model_validator(mode="before")
    @classmethod
    def reject_mixed_provider_aliases(cls, value: Any) -> Any:
        if isinstance(value, dict) and "provider" in value and "custom" in value:
            raise ValueError("env.rewards cannot define both provider and legacy custom")
        return value

    @field_validator("provider", mode="before")
    @classmethod
    def expand_provider_shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value

    @property
    def custom(self) -> RewardSelector | None:
        """Deprecated Python compatibility alias for ``provider``."""
        return self.provider

    @model_validator(mode="after")
    def validate_zone_rewards(self) -> "RewardConfig":
        if self.zone_reward_min > self.zone_reward_max:
            raise ValueError("zone_reward_min must be less than or equal to zone_reward_max")
        if self.zone_reward_max >= 0.0:
            raise ValueError("zone_reward_max must stay negative")
        if self.zone_reward_min >= 0.0:
            raise ValueError("zone_reward_min must stay negative")
        return self


