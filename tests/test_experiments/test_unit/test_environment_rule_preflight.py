"""Tests for environment-rule registration and pre-Ray validation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from theseo_anysearch.environment_rules import (
    EnvironmentRuleMetadata,
    EnvironmentRulePreflightError,
    EnvironmentRuleRegistry,
    RuleReference,
    built_in_environment_rule_registry,
    preflight_environment_rules,
)
from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.settings.environment.action import ActionConfig


def _with_action(
    config: ExperimentConfig,
    action: ActionConfig,
) -> ExperimentConfig:
    env = config.env.model_copy(update={"action": action})
    return config.model_copy(update={"env": env})


def test_default_action_pipeline_passes_preflight(
    experiment_config: ExperimentConfig,
) -> None:
    registry = preflight_environment_rules(experiment_config)

    assert registry.names("predicate") == ("bounds", "unoccupied", "valid_action")
    assert "trail_placement" in registry.names("outcome")


def test_unknown_rule_reports_exact_yaml_path(
    experiment_config: ExperimentConfig,
) -> None:
    config = _with_action(
        experiment_config,
        ActionConfig(predicates=("valid_action", "unknown_rule")),
    )

    with pytest.raises(
        EnvironmentRulePreflightError,
        match=r"env\.action\.predicates\[1\].*unknown predicate 'unknown_rule'",
    ):
        preflight_environment_rules(config)


def test_missing_dependency_is_rejected(
    experiment_config: ExperimentConfig,
) -> None:
    config = _with_action(
        experiment_config,
        ActionConfig(predicates=("bounds",)),
    )

    with pytest.raises(
        EnvironmentRulePreflightError,
        match="predicate 'bounds' requires predicate 'valid_action'",
    ):
        preflight_environment_rules(config)


def test_dependency_order_is_rejected(
    experiment_config: ExperimentConfig,
) -> None:
    config = _with_action(
        experiment_config,
        ActionConfig(predicates=("bounds", "valid_action")),
    )

    with pytest.raises(
        EnvironmentRulePreflightError,
        match="'valid_action' must appear before predicate 'bounds'",
    ):
        preflight_environment_rules(config)


def test_registered_conflicting_rule_is_rejected(
    experiment_config: ExperimentConfig,
) -> None:
    registry = built_in_environment_rule_registry()
    registry.register(
        EnvironmentRuleMetadata(
            name="stationary",
            kind="outcome",
            source="python",
            conflicts=(RuleReference(kind="outcome", name="cursor_movement"),),
        )
    )
    config = _with_action(
        experiment_config,
        ActionConfig(outcomes=("cursor_movement", "stationary")),
    )

    with pytest.raises(
        EnvironmentRulePreflightError,
        match="outcome 'stationary' conflicts with outcome 'cursor_movement'",
    ):
        preflight_environment_rules(config, registry=registry)


def test_python_reward_is_resolved_before_runner_start(
    experiment_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    config_path = tmp_path.joinpath("experiment.yaml")
    config_path.write_text("description: test\n", encoding="utf-8")
    tmp_path.joinpath("rewards.py").write_text(
        "def sparse_goal(context):\n    return {'reward': 0.0}\n",
        encoding="utf-8",
    )
    rewards = type(experiment_config.env.rewards).model_validate(
        {"provider": {"name": "sparse_goal"}}
    )
    config = experiment_config.model_copy(
        update={"env": experiment_config.env.model_copy(update={"rewards": rewards})}
    )

    registry = preflight_environment_rules(config, config_path)

    assert registry.require("reward", "sparse_goal").source == "python"


def test_unknown_reward_fails_without_silent_fallback(
    experiment_config: ExperimentConfig,
) -> None:
    rewards = type(experiment_config.env.rewards).model_validate(
        {"provider": {"name": "missing_reward"}}
    )
    config = experiment_config.model_copy(
        update={"env": experiment_config.env.model_copy(update={"rewards": rewards})}
    )

    with pytest.raises(
        EnvironmentRulePreflightError,
        match="env.rewards.provider: unknown reward 'missing_reward'",
    ):
        preflight_environment_rules(config)


def test_structured_native_metadata_drives_dependency_validation(
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from theseo_anysearch.environment_rules import preflight as preflight_module

    native_guard = EnvironmentRuleMetadata(
        name="native_guard",
        kind="predicate",
        version=3,
        source="native",
        environment_families=frozenset({"voxel", "surface"}),
        dependencies=(RuleReference(kind="predicate", name="valid_action"),),
    )
    manifest = SimpleNamespace(
        rule_metadata=(native_guard,),
        predicates=("native_guard",),
        outcomes=(),
        rewards=(),
    )
    monkeypatch.setattr(preflight_module, "_native_manifest", lambda _: manifest)
    config = _with_action(
        experiment_config,
        ActionConfig(predicates=("valid_action", "native_guard")),
    )

    registry = preflight_environment_rules(config, Path("experiment.yaml"))

    assert registry.require("predicate", "native_guard") == native_guard


def test_incompatible_native_metadata_schema_fails_deterministically(
    experiment_config: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from theseo_anysearch.experiments import native_extensions

    manifest_path = tmp_path.joinpath("extension.json")
    manifest_path.write_text(
        """{
          "abi_version": 2,
          "source_sha256": "source",
          "binary_sha256": "binary",
          "library": "extension.dll",
          "capabilities": [],
          "rule_metadata_schema_version": 99,
          "rule_metadata": [],
          "platform": "win32",
          "machine": "AMD64"
        }""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        native_extensions,
        "discover_native_manifest",
        lambda _: manifest_path,
    )

    with pytest.raises(
        EnvironmentRulePreflightError,
        match="native extension rule metadata is incompatible.*Input should be 1",
    ):
        preflight_environment_rules(experiment_config, tmp_path.joinpath("experiment.yaml"))


def test_name_only_native_manifest_remains_compatible() -> None:
    from theseo_anysearch.experiments.native_extensions import NativeExtensionManifest

    manifest = NativeExtensionManifest.model_validate(
        {
            "abi_version": 2,
            "source_sha256": "source",
            "binary_sha256": "binary",
            "library": "extension.dll",
            "capabilities": ["predicate"],
            "predicates": ["legacy_guard"],
            "platform": "win32",
            "machine": "AMD64",
        }
    )

    assert manifest.rule_metadata_schema_version == 1
    assert manifest.rule_metadata == ()


def test_rule_metadata_requires_an_environment_family() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        EnvironmentRuleMetadata(
            name="nowhere",
            kind="predicate",
            environment_families=frozenset(),
        )
