"""Resolve environment rules and reject invalid YAML before Ray starts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from theseo_anysearch.environment_rules.models import EnvironmentRuleMetadata, RuleKind
from theseo_anysearch.environment_rules.registry import (
    EnvironmentRuleRegistry,
    built_in_environment_rule_registry,
)

if TYPE_CHECKING:
    from theseo_anysearch.experiments.models import ExperimentConfig
    from theseo_anysearch.settings.environment.action import ActionExtensionSelector


class EnvironmentRulePreflightError(ValueError):
    """Raised when configured environment rules cannot compose safely."""


def _native_manifest(config_path: Path | None):
    if config_path is None:
        return None
    from theseo_anysearch.experiments.native_extensions import (
        NativeExtensionManifest,
        discover_native_manifest,
    )

    manifest_path = discover_native_manifest(config_path)
    if manifest_path is None:
        return None
    return NativeExtensionManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )


def _register_native_rules(
    registry: EnvironmentRuleRegistry,
    config_path: Path | None,
) -> None:
    manifest = _native_manifest(config_path)
    if manifest is None:
        return
    for kind, names in (
        ("predicate", manifest.predicates),
        ("outcome", manifest.outcomes),
        ("reward", manifest.rewards),
    ):
        for name in names:
            registry.register(
                EnvironmentRuleMetadata(
                    name=name,
                    kind=kind,
                    version=manifest.abi_version,
                    source="native",
                ),
                replace_identical_name=True,
            )


def _register_python_reward(
    registry: EnvironmentRuleRegistry,
    config_path: Path | None,
    reward_name: str | None,
) -> None:
    if config_path is None or reward_name is None:
        return
    from theseo_anysearch.experiments.custom_rewards import (
        discover_reward_source,
        load_reward_provider,
    )

    source = discover_reward_source(config_path, reward_name)
    provider = load_reward_provider(source, reward_name)
    if provider is not None:
        registry.register(
            EnvironmentRuleMetadata(
                name=reward_name,
                kind="reward",
                source="python",
            ),
            replace_identical_name=True,
        )


def _selected_pipeline(
    config: "ExperimentConfig",
) -> Iterable[tuple[str, RuleKind, tuple["ActionExtensionSelector", ...]]]:
    predicates, outcomes = config.env.action.resolved_pipeline(
        trail_mode=config.env.trail_mode
    )
    yield "env.action.predicates", "predicate", predicates
    yield "env.action.outcomes", "outcome", outcomes
    for index, agent in enumerate(config.env.agents or ()):
        predicates, outcomes = agent.action.resolved_pipeline(trail_mode=False)
        yield f"env.agents[{index}].action.predicates", "predicate", predicates
        yield f"env.agents[{index}].action.outcomes", "outcome", outcomes


def _validate_pipeline(
    registry: EnvironmentRuleRegistry,
    path: str,
    kind: RuleKind,
    selectors: tuple["ActionExtensionSelector", ...],
    *,
    environment_family: str,
    selected_rules: set[tuple[str, str]],
) -> None:
    positions = {(kind, selector.name): index for index, selector in enumerate(selectors)}
    for index, selector in enumerate(selectors):
        try:
            rule = registry.require(kind, selector.name)
        except KeyError as exc:
            raise EnvironmentRulePreflightError(f"{path}[{index}]: {exc.args[0]}") from None
        if environment_family not in rule.environment_families:
            raise EnvironmentRulePreflightError(
                f"{path}[{index}]: {kind} {selector.name!r} does not support "
                f"the {environment_family!r} environment family"
            )
        for dependency in rule.dependencies:
            if dependency.key not in selected_rules:
                raise EnvironmentRulePreflightError(
                    f"{path}[{index}]: {kind} {selector.name!r} requires "
                    f"{dependency.kind} {dependency.name!r}"
                )
            if dependency.key in positions and positions[dependency.key] >= index:
                raise EnvironmentRulePreflightError(
                    f"{path}[{index}]: {dependency.kind} {dependency.name!r} "
                    f"must appear before {kind} {selector.name!r}"
                )
        for conflict in rule.conflicts:
            if conflict.key in selected_rules:
                raise EnvironmentRulePreflightError(
                    f"{path}[{index}]: {kind} {selector.name!r} conflicts with "
                    f"{conflict.kind} {conflict.name!r}"
                )


def preflight_environment_rules(
    config: "ExperimentConfig",
    config_path: Path | None = None,
    *,
    registry: EnvironmentRuleRegistry | None = None,
) -> EnvironmentRuleRegistry:
    """Resolve all selected rules before any Ray trainer or worker is created."""
    resolved = registry or built_in_environment_rule_registry()
    reward_name = (
        config.env.rewards.provider.name if config.env.rewards.provider is not None else None
    )
    _register_python_reward(resolved, config_path, reward_name)
    # Native implementations supersede Python implementations only for the
    # identical typed name, matching runtime reward and action resolution.
    _register_native_rules(resolved, config_path)

    pipelines = tuple(_selected_pipeline(config))
    selected_rules = {
        (kind, selector.name)
        for _, kind, selectors in pipelines
        for selector in selectors
    }
    if reward_name is not None:
        selected_rules.add(("reward", reward_name))
    for path, kind, selectors in pipelines:
        _validate_pipeline(
            resolved,
            path,
            kind,
            selectors,
            environment_family="voxel",
            selected_rules=selected_rules,
        )

    if reward_name is not None and resolved.get("reward", reward_name) is None:
        location = "env.rewards.provider"
        extension_hint = ""
        if config_path is not None and config_path.parent.joinpath("extension").is_dir():
            extension_hint = f"; run 'anysearch compile {config_path.parent}'"
        raise EnvironmentRulePreflightError(
            f"{location}: unknown reward {reward_name!r}{extension_hint}"
        )
    if config.staging is not None and config.staging.enabled:
        for index in range(len(config.staging.stages)):
            stage_config = config.stage_experiment(index, completed_iterations=0)
            try:
                preflight_environment_rules(
                    stage_config,
                    config_path,
                    registry=resolved,
                )
            except EnvironmentRulePreflightError as exc:
                raise EnvironmentRulePreflightError(
                    f"staging.stages[{index}]: {exc}"
                ) from None
    return resolved
