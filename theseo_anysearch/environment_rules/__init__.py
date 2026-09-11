"""Public environment-rule metadata, registration, and preflight APIs."""

from theseo_anysearch.environment_rules.models import (
    EnvironmentFamily,
    EnvironmentRuleMetadata,
    RuleKind,
    RuleReference,
)
from theseo_anysearch.environment_rules.preflight import (
    EnvironmentRulePreflightError,
    preflight_environment_rules,
)
from theseo_anysearch.environment_rules.registry import (
    EnvironmentRuleRegistry,
    built_in_environment_rule_registry,
    register_environment_rule,
)

__all__ = [
    "EnvironmentFamily",
    "EnvironmentRuleMetadata",
    "EnvironmentRulePreflightError",
    "EnvironmentRuleRegistry",
    "RuleKind",
    "RuleReference",
    "built_in_environment_rule_registry",
    "preflight_environment_rules",
    "register_environment_rule",
]
