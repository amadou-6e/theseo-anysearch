"""Registration and discovery for composable environment rules."""

from collections.abc import Iterable

from theseo_anysearch.environment_rules.models import (
    EnvironmentRuleMetadata,
    RuleKind,
    RuleReference,
)

_THIRD_PARTY_RULES: dict[tuple[str, str], EnvironmentRuleMetadata] = {}


class EnvironmentRuleRegistry:
    """Store typed rule metadata and reject ambiguous registrations."""

    def __init__(self, rules: Iterable[EnvironmentRuleMetadata] = ()) -> None:
        self._rules: dict[tuple[str, str], EnvironmentRuleMetadata] = {}
        for rule in rules:
            self.register(rule)

    def register(
        self,
        rule: EnvironmentRuleMetadata,
        *,
        replace_identical_name: bool = False,
    ) -> None:
        """Register a rule, optionally superseding the same typed name."""
        current = self._rules.get(rule.key)
        if current is not None and not replace_identical_name:
            raise ValueError(
                f"duplicate {rule.kind} registration {rule.name!r}: "
                f"{current.source} and {rule.source}"
            )
        self._rules[rule.key] = rule

    def get(self, kind: RuleKind, name: str) -> EnvironmentRuleMetadata | None:
        """Return metadata for a typed name, if registered."""
        return self._rules.get((kind, name))

    def require(self, kind: RuleKind, name: str) -> EnvironmentRuleMetadata:
        """Return metadata or raise an actionable unknown-rule error."""
        rule = self.get(kind, name)
        if rule is None:
            available = ", ".join(self.names(kind)) or "<none>"
            raise KeyError(
                f"unknown {kind} {name!r}; registered {kind} names: {available}"
            )
        return rule

    def names(self, kind: RuleKind) -> tuple[str, ...]:
        """Return sorted names registered for one rule kind."""
        return tuple(sorted(name for rule_kind, name in self._rules if rule_kind == kind))

    def metadata(self) -> tuple[EnvironmentRuleMetadata, ...]:
        """Return deterministic metadata suitable for diagnostics and tooling."""
        return tuple(self._rules[key] for key in sorted(self._rules))


def _reference(kind: RuleKind, name: str) -> RuleReference:
    return RuleReference(kind=kind, name=name)


def built_in_environment_rule_registry() -> EnvironmentRuleRegistry:
    """Return a fresh registry containing AnySearch built-in voxel rules."""
    valid_action = _reference("predicate", "valid_action")
    bounds = _reference("predicate", "bounds")
    cursor_movement = _reference("outcome", "cursor_movement")
    registry = EnvironmentRuleRegistry(
        (
            EnvironmentRuleMetadata(name="valid_action", kind="predicate"),
            EnvironmentRuleMetadata(
                name="bounds", kind="predicate", dependencies=(valid_action,)
            ),
            EnvironmentRuleMetadata(
                name="unoccupied", kind="predicate", dependencies=(bounds,)
            ),
            EnvironmentRuleMetadata(name="cursor_movement", kind="outcome"),
            EnvironmentRuleMetadata(
                name="trail_placement",
                kind="outcome",
                dependencies=(cursor_movement,),
            ),
            EnvironmentRuleMetadata(name="place", kind="outcome"),
            EnvironmentRuleMetadata(name="remove", kind="outcome"),
        )
    )
    for rule in _THIRD_PARTY_RULES.values():
        registry.register(rule, replace_identical_name=True)
    return registry


def register_environment_rule(rule: EnvironmentRuleMetadata) -> None:
    """Register third-party metadata used by subsequent runner preflights."""
    current = _THIRD_PARTY_RULES.get(rule.key)
    if current is not None:
        raise ValueError(
            f"duplicate third-party {rule.kind} registration {rule.name!r}"
        )
    _THIRD_PARTY_RULES[rule.key] = rule
