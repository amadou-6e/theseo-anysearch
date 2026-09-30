"""Versioned metadata contracts for composable environment rules."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RuleKind = Literal[
    "predicate",
    "outcome",
    "reward",
    "training_metrics",
    "evaluation_metrics",
    "scenario",
]
EnvironmentFamily = Literal["voxel", "surface"]


class RuleReference(BaseModel):
    """Identify another rule by kind and selectable name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RuleKind
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable registry key."""
        return self.kind, self.name


class EnvironmentRuleMetadata(BaseModel):
    """Describe one selectable environment rule without constructing a trainer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    kind: RuleKind
    version: int = Field(default=1, ge=1)
    source: Literal["built_in", "python", "native"] = "built_in"
    environment_families: frozenset[EnvironmentFamily] = Field(
        default_factory=lambda: frozenset({"voxel"}), min_length=1
    )
    dependencies: tuple[RuleReference, ...] = ()
    conflicts: tuple[RuleReference, ...] = ()

    @model_validator(mode="after")
    def validate_relationships(self) -> "EnvironmentRuleMetadata":
        """Reject duplicated and self-referential relationships."""
        for label, references in (
            ("dependencies", self.dependencies),
            ("conflicts", self.conflicts),
        ):
            keys = [reference.key for reference in references]
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate {label} for {self.kind} {self.name!r}")
            if (self.kind, self.name) in keys:
                raise ValueError(f"{self.kind} {self.name!r} cannot reference itself")
        return self

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable registry key."""
        return self.kind, self.name
