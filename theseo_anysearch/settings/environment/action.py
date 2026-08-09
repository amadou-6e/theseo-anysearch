"""Policy action-space settings."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class ActionExtensionSelector(BaseModel):
    """Select a built-in or compiled Rust action extension by name."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ActionMaskingConfig(BaseModel):
    """Expose predicate feasibility to discrete policies as an action mask."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    all_masked: Literal["error"] = "error"


class ActionConfig(BaseModel):
    """Policy action-space and Rust behavior pipeline."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["discrete_6", "discrete_18", "discrete_26", "vector_3"] = Field(
        "discrete_26", description="Movement action space exposed to the policy."
    )
    behavior: Literal["legacy", "cursor_navigation", "trail_navigation"] = Field(
        "legacy", description="Built-in action predicate and outcome preset."
    )
    predicates: tuple[ActionExtensionSelector, ...] | None = Field(
        None, description="Ordered action-feasibility predicate selectors."
    )
    outcomes: tuple[ActionExtensionSelector, ...] | None = Field(
        None, description="Ordered post-action environment outcome selectors."
    )
    history_length: int = Field(
        default=16, ge=0, le=4096, description="Actions retained for extension context."
    )
    masking: ActionMaskingConfig = Field(default_factory=ActionMaskingConfig)

    @field_validator("predicates", "outcomes", mode="before")
    @classmethod
    def expand_selector_shorthand(cls, value: Any) -> Any:
        """Expand string selectors into their object representation."""
        if value is None:
            return None
        return [({"name": item} if isinstance(item, str) else item) for item in value]

    @model_validator(mode="after")
    def validate_unique_pipeline_names(self) -> "ActionConfig":
        """Reject duplicate extension names within either pipeline."""
        for kind, selectors in (("predicate", self.predicates), ("outcome", self.outcomes)):
            if selectors is None:
                continue
            names = [selector.name for selector in selectors]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"duplicate action {kind} names: {duplicates}")
        if self.masking.enabled and self.mode == "vector_3":
            raise ValueError(
                "action masking requires a discrete action mode; vector_3 uses "
                "factorized MultiDiscrete logits that cannot represent an arbitrary "
                "joint predicate mask"
            )
        return self

    def resolved_pipeline(
        self, *, trail_mode: bool
    ) -> tuple[tuple[ActionExtensionSelector, ...], tuple[ActionExtensionSelector, ...]]:
        """Resolve legacy behavior into explicit predicate and outcome pipelines."""
        behavior = self.behavior
        if behavior == "legacy":
            behavior = "trail_navigation" if trail_mode else "cursor_navigation"
        predicates = self.predicates if self.predicates is not None else tuple(
            ActionExtensionSelector(name=name)
            for name in ("valid_action", "bounds", "unoccupied")
        )
        default_outcomes = (
            ("cursor_movement", "trail_placement")
            if behavior == "trail_navigation"
            else ("cursor_movement",)
        )
        outcomes = self.outcomes if self.outcomes is not None else tuple(
            ActionExtensionSelector(name=name) for name in default_outcomes
        )
        return predicates, outcomes
