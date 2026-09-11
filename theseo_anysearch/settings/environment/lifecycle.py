"""Episode lifecycle rule selection."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class LifecycleRuleSelector(BaseModel):
    """Select one registered lifecycle rule and provide its parameters."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class LifecycleConfig(BaseModel):
    """Ordered rules used to interpret Rust transitions as episode outcomes."""

    model_config = ConfigDict(extra="forbid")
    rules: tuple[LifecycleRuleSelector, ...] = (
        LifecycleRuleSelector(name="native"),
    )

    @field_validator("rules", mode="before")
    @classmethod
    def expand_rule_shorthand(cls, value: Any) -> Any:
        return [({"name": item} if isinstance(item, str) else item) for item in value]

    @model_validator(mode="after")
    def validate_rules(self) -> "LifecycleConfig":
        if not self.rules:
            raise ValueError("env.lifecycle.rules must contain at least one rule")
        names = [rule.name for rule in self.rules]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate lifecycle rule names: {duplicates}")
        return self
