"""Per-agent settings for heterogeneous multi-agent environments."""

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.settings.environment.action import ActionConfig


class AgentConfig(BaseModel):
    """Configuration that may differ between agents in one environment."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    action: ActionConfig = Field(default_factory=ActionConfig)
    start: tuple[int, int, int] | None = Field(
        None, description="Optional fixed reset position."
    )
    policy: str | None = Field(
        None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="RLlib policy name; defaults to the agent id.",
    )


class HunterAndHuntedConfig(BaseModel):
    """Asymmetric capture task for two configured agents."""

    model_config = ConfigDict(extra="forbid")

    hunter: str
    hunted: str
    capture_distance: int = Field(1, ge=0)
    hunter_capture_reward: float = 1.0
    hunted_escape_reward: float = 1.0
