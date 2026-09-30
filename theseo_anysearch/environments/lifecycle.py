"""Composable, deterministic interpretation of environment transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class LifecycleContext(BaseModel):
    """Immutable state exposed after Rust has applied an action."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    step: int
    action: Any
    action_index: int
    cursor: tuple[int, int, int]
    goal: tuple[int, int, int] | None
    goal_distance: float
    collision: bool
    invalid_action: bool
    native_success: bool
    native_terminated: bool
    native_truncated: bool
    native_reason: str
    route_complete: bool
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)


class LifecycleDecision(BaseModel):
    """Fields set by one rule; omitted fields retain the previous decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool | None = None
    failure: bool | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    reason: str | None = None
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reason(self) -> "LifecycleDecision":
        if self.reason is not None and not self.reason:
            raise ValueError("lifecycle reason cannot be empty")
        return self


class LifecycleOutcome(BaseModel):
    """Resolved Gymnasium outcome and typed diagnostic information."""

    model_config = ConfigDict(frozen=True)

    success: bool
    failure: bool
    terminated: bool
    truncated: bool
    reason: str
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)


class LifecycleRule(Protocol):
    """A read-only transition rule. Rust remains responsible for mutation."""

    def __call__(self, context: LifecycleContext) -> LifecycleDecision: ...


LifecycleRuleFactory = Callable[[Mapping[str, JsonValue]], LifecycleRule]
_RULE_FACTORIES: dict[str, LifecycleRuleFactory] = {}


def register_lifecycle_rule(
    name: str, factory: LifecycleRuleFactory, *, replace: bool = False
) -> None:
    """Register a named rule factory before constructing an environment."""
    if not name.isidentifier():
        raise ValueError("lifecycle rule name must be a Python identifier")
    if name in _RULE_FACTORIES and not replace:
        raise ValueError(f"lifecycle rule {name!r} is already registered")
    _RULE_FACTORIES[name] = factory


def _native_rule(parameters: Mapping[str, JsonValue]) -> LifecycleRule:
    if parameters:
        raise ValueError("native lifecycle rule does not accept parameters")

    def evaluate(context: LifecycleContext) -> LifecycleDecision:
        success = context.native_success and context.route_complete
        failure = context.native_terminated and not context.native_success
        return LifecycleDecision(
            success=success,
            failure=failure,
            terminated=context.native_terminated if context.route_complete else failure,
            truncated=context.native_truncated,
            reason=context.native_reason if context.route_complete else "in_progress",
        )

    return evaluate


register_lifecycle_rule("native", _native_rule)


def build_lifecycle_rules(selectors: Sequence[Mapping[str, Any]]) -> tuple[LifecycleRule, ...]:
    """Instantiate YAML-selected rules in their declared order."""
    rules: list[LifecycleRule] = []
    for selector in selectors:
        name = str(selector["name"])
        factory = _RULE_FACTORIES.get(name)
        if factory is None:
            available = ", ".join(sorted(_RULE_FACTORIES))
            raise ValueError(f"unknown lifecycle rule {name!r}; registered rules: {available}")
        rules.append(factory(dict(selector.get("parameters") or {})))
    return tuple(rules)


def evaluate_lifecycle(
    rules: Sequence[LifecycleRule], context: LifecycleContext
) -> LifecycleOutcome:
    """Apply ordered overrides, then enforce one deterministic terminal state."""
    state: dict[str, Any] = {
        "success": False,
        "failure": False,
        "terminated": False,
        "truncated": False,
        "reason": "in_progress",
    }
    diagnostics: dict[str, JsonValue] = {}
    for rule in rules:
        decision = LifecycleDecision.model_validate(rule(context))
        for field in ("success", "failure", "terminated", "truncated", "reason"):
            value = getattr(decision, field)
            if value is not None:
                state[field] = value
        overlap = diagnostics.keys() & decision.diagnostics.keys()
        if overlap:
            raise ValueError(
                f"lifecycle diagnostic keys must be unique across rules: {sorted(overlap)}"
            )
        diagnostics.update(decision.diagnostics)

    if state["success"] and state["failure"]:
        raise ValueError("lifecycle cannot resolve both success and failure")
    if state["success"] or state["failure"]:
        state["terminated"] = True
    if state["terminated"]:
        state["truncated"] = False
    if not state["terminated"] and not state["truncated"]:
        state["reason"] = "in_progress"
    return LifecycleOutcome(**state, diagnostics=diagnostics)
