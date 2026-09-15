"""Episode scenario-provider settings."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class ScenarioProviderSelector(BaseModel):
    """Select a named Python or Rust episode scenario provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    """Scenario generation applied immediately before an environment reset."""

    model_config = ConfigDict(extra="forbid")

    provider: ScenarioProviderSelector | None = None
    candidate_index: Path | None = Field(
        default=None,
        description="Compiled-world directory containing candidates.idx and candidates.bin.",
    )
    maximum_candidate_queries: int = Field(default=64, ge=1)
    maximum_candidate_results: int = Field(default=4096, ge=1)

    @field_validator("provider", mode="before")
    @classmethod
    def expand_provider_shorthand(cls, value: Any) -> Any:
        """Accept ``provider: name`` as shorthand for the selector block."""
        if isinstance(value, str):
            return {"name": value}
        return value
