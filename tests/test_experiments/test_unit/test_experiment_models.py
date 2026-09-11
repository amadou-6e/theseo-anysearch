"""Unit tests for experiment model and loader behavior."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
from theseo_anysearch.models import EnvConfig
from theseo_anysearch.experiments.models import (
    ExperimentConfig,
    HeuristicConfig,
    SweepConfig,
)


class TestExperimentModels:
    """Verify typed experiment loading and model conversions."""

    def test_loads_single_yaml(self, single_yaml: Path):
        result = load_experiment(single_yaml)
        assert isinstance(result, ExperimentConfig)

    def test_experiment_name(self, experiment_config: ExperimentConfig):
        assert experiment_config.experiment.name == "test-run"

    def test_algorithm_config_typed(self, experiment_config: ExperimentConfig):
        from theseo_anysearch.rllib.algorithms.models import PPOConfig

        assert isinstance(experiment_config.algorithm_config, PPOConfig)

    def test_model_cfg_typed(self, experiment_config: ExperimentConfig):
        from theseo_anysearch.rllib.models.models import VoxelEncoderConfig

        assert isinstance(experiment_config.model_cfg, VoxelEncoderConfig)

    def test_run_output_dir_includes_name(self, experiment_config: ExperimentConfig):
        assert experiment_config.run_output_dir.name == "test-run"

    def test_renders_default_empty(self, experiment_config: ExperimentConfig):
        assert experiment_config.renders.camera_positions == []

    def test_tune_config_absent(self, experiment_config: ExperimentConfig):
        assert experiment_config.tune_config is None

    def test_loads_sweep_yaml(self, sweep_yaml: Path):
        result = load_experiment(sweep_yaml)
        assert isinstance(result, SweepConfig)

    def test_expand_sweep_produces_two_experiments(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        assert isinstance(sweep, SweepConfig)
        entries = expand_sweep(sweep)
        assert len(entries) == 2

    def test_expand_sweep_names(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        names = [entry.experiment.name for entry in entries]
        assert "sweep-lr-low" in names
        assert "sweep-lr-high" in names

    def test_expand_sweep_overrides_lr(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        lrs = {entry.experiment.name: entry.algorithm_config.lr for entry in entries}
        assert lrs["sweep-lr-low"] == pytest.approx(1e-4)
        assert lrs["sweep-lr-high"] == pytest.approx(1e-2)

    def test_to_settings_returns_settings(self, experiment_config: ExperimentConfig):
        from theseo_anysearch.models import Settings

        settings = experiment_config.to_settings()
        assert isinstance(settings, Settings)

    def test_anyscale_config_preserved_when_present(self, tmp_path: Path):
        yaml_text = textwrap.dedent(f"""\
            experiment:
              name: anyscale-run
              output_dir: {tmp_path}

            env:
              stl_path: /tmp/test.stl
              agent_count: 1

            training:
              algorithm: ppo
              runner: anyscale

            anyscale:
              cluster_env: env-1
              compute_config: compute-1
              project: proj-1
        """)
        path = tmp_path.joinpath("experiment_anyscale.yaml")
        path.write_text(yaml_text, encoding="utf-8")

        result = load_experiment(path)
        assert isinstance(result, ExperimentConfig)
        assert result.to_settings().anyscale.project == "proj-1"


class TestStagingConfig:
    def _payload(self, experiment_config: ExperimentConfig) -> dict:
        return experiment_config.model_dump(by_alias=True, mode="python")

    def test_resolves_ordered_environment_overrides(
        self,
        experiment_config: ExperimentConfig,
    ):
        payload = self._payload(experiment_config)
        payload["staging"] = {
            "stages": [
                {
                    "name": "select-goal",
                    "completion": {"type": "iterations", "iterations": 2},
                    "env": {"max_steps": 1, "trail_mode": False},
                },
                {
                    "name": "full-route",
                    "completion": {"type": "iterations", "iterations": 3},
                    "env": {"max_steps": 50, "trail_mode": True},
                },
            ]
        }
        config = ExperimentConfig.model_validate(payload)

        first = config.stage_experiment(0, completed_iterations=0)
        second = config.stage_experiment(1, completed_iterations=2)

        assert first.env.max_steps == 1
        assert first.env.trail_mode is False
        assert first.training.iterations == 2
        assert second.env.max_steps == 50
        assert second.env.trail_mode is True
        assert second.training.iterations == 5

    def test_rejects_duplicate_stage_names(
        self,
        experiment_config: ExperimentConfig,
    ):
        payload = self._payload(experiment_config)
        payload["staging"] = {
            "stages": [
                {"name": "same", "completion": {"type": "iterations", "iterations": 1}},
                {"name": "same", "completion": {"type": "iterations", "iterations": 1}},
            ]
        }

        with pytest.raises(ValueError, match="unique"):
            ExperimentConfig.model_validate(payload)

    @pytest.mark.parametrize(
        "env_override",
        [
            {"observation": {"mode": "radial"}},
            {"action": {"mode": "discrete_6"}},
            {"geometry": {"grid_size": 64}},
            {"agent_count": 2},
        ],
    )
    def test_rejects_policy_contract_changes(
        self,
        experiment_config: ExperimentConfig,
        env_override: dict,
    ):
        payload = self._payload(experiment_config)
        payload["staging"] = {
            "stages": [
                {"name": "invalid", "completion": {"type": "iterations", "iterations": 1}, "env": env_override}
            ]
        }

        with pytest.raises(ValueError, match="policy-contract"):
            ExperimentConfig.model_validate(payload)

    def test_stage_overrides_are_recursive_and_base_relative(
        self,
        experiment_config: ExperimentConfig,
    ):
        payload = self._payload(experiment_config)
        base_runners = payload["evaluation"]["num_env_runners"]
        base_episodes = payload["evaluation"]["episodes"]
        payload["staging"] = {
            "stages": [
                {
                    "name": "first",
                    "completion": {"type": "iterations", "iterations": 1},
                    "evaluation": {"num_env_runners": base_runners + 1},
                    "training": {"checkpoint_interval": 1},
                },
                {
                    "name": "second",
                    "completion": {"type": "iterations", "iterations": 1},
                    "evaluation": {"episodes": base_episodes + 1},
                },
            ]
        }

        config = ExperimentConfig.model_validate(payload)
        first = config.stage_experiment(0, completed_iterations=0)
        second = config.stage_experiment(1, completed_iterations=1)

        assert first.evaluation.num_env_runners == base_runners + 1
        assert first.evaluation.episodes == base_episodes
        assert first.training.checkpoint_interval == 1
        assert second.evaluation.num_env_runners == base_runners
        assert second.evaluation.episodes == base_episodes + 1

    @pytest.mark.parametrize(
        ("block", "override"),
        [
            ("evaluation", {"episodes": 0}),
            ("algorithm_config", {"lr": -1.0}),
            ("training", {"num_env_runners": -1}),
        ],
    )
    def test_invalid_future_stage_fails_during_root_validation(
        self,
        experiment_config: ExperimentConfig,
        block: str,
        override: dict,
    ):
        payload = self._payload(experiment_config)
        payload["staging"] = {
            "stages": [{
                "name": "invalid-later",
                "completion": {"type": "iterations", "iterations": 1},
                block: override,
            }]
        }

        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(payload)

    def test_rejects_replay_preservation_for_dqn(
        self,
        experiment_config: ExperimentConfig,
    ):
        payload = self._payload(experiment_config)
        payload["training"]["algorithm"] = "dqn"
        payload["training"]["model"] = "fcnet"
        payload["algorithm_config"] = {"lr": 0.001}
        payload["staging"] = {
            "replay_transition": "preserve",
            "stages": [{
                "name": "invalid-replay",
                "completion": {"type": "iterations", "iterations": 1},
            }],
        }

        with pytest.raises(ValueError, match="replay-buffer preservation"):
            ExperimentConfig.model_validate(payload)

    def test_max_iterations_requires_explicit_policy(self):
        with pytest.raises(ValueError, match="on_max_iterations"):
            from theseo_anysearch.experiments.models import StageCompletionConfig

            StageCompletionConfig(
                type="performance",
                metric="evaluation_success_rate",
                threshold=0.9,
                max_iterations=10,
            )

class TestCustomRewardConfig:
    def test_string_shorthand_preserves_existing_yaml(self):
        config = EnvConfig.model_validate({"rewards": {"custom": "my_reward"}})

        assert config.rewards.custom is not None
        assert config.rewards.custom.name == "my_reward"
        assert config.rewards.custom.parameters == {}
        assert config.to_runtime_dict()["custom_reward"] == "my_reward"

    def test_structured_reward_exposes_json_parameters(self):
        config = EnvConfig.model_validate({
            "rewards": {
                "custom": {
                    "name": "my_reward",
                    "parameters": {
                        "scale": 2.5,
                        "enabled": True,
                        "labels": ["fast", "sparse"],
                    },
                }
            }
        })

        runtime = config.to_runtime_dict()
        assert runtime["custom_reward"] == "my_reward"
        assert runtime["custom_reward_parameters"] == {
            "scale": 2.5,
            "enabled": True,
            "labels": ["fast", "sparse"],
        }

class TestHeuristicConfig:
    """Verify YAML-facing heuristic configuration and compatibility."""

    def test_defaults_to_disabled_astar(self):
        config = HeuristicConfig()

        assert config.enabled is False
        assert config.type == "astar"
        assert config.weight is None

    def test_weighted_astar_defaults_weight(self):
        config = HeuristicConfig(enabled=True, type="weighted_astar")

        assert config.weight == pytest.approx(1.5)

    @pytest.mark.parametrize("weight", [0.0, -1.0])
    def test_weighted_astar_requires_positive_weight(self, weight):
        with pytest.raises(ValueError, match="greater than zero"):
            HeuristicConfig(
                enabled=True,
                type="weighted_astar",
                weight=weight,
            )

    def test_weight_is_rejected_for_other_types(self):
        with pytest.raises(ValueError, match="only valid"):
            HeuristicConfig(enabled=True, type="dijkstra", weight=2.0)

    def test_enabled_heuristic_requires_single_agent(
        self,
        experiment_config: ExperimentConfig,
    ):
        payload = experiment_config.model_dump(by_alias=True, mode="python")
        payload["env"]["agent_count"] = 2
        payload["heuristic"] = {"enabled": True, "type": "astar"}

        with pytest.raises(ValueError, match="agent_count"):
            ExperimentConfig.model_validate(payload)
    def test_standalone_heuristic_requires_enabled_config(
        self,
        experiment_config: ExperimentConfig,
    ):
        payload = experiment_config.model_dump(by_alias=True, mode="python")
        payload["training"]["algorithm"] = "heuristic"

        with pytest.raises(ValueError, match="enabled"):
            ExperimentConfig.model_validate(payload)

    def test_standalone_heuristic_accepts_local_single_agent(
        self,
        experiment_config: ExperimentConfig,
    ):
        payload = experiment_config.model_dump(by_alias=True, mode="python")
        payload["training"]["algorithm"] = "heuristic"
        payload["heuristic"] = {"enabled": True, "type": "dijkstra"}

        config = ExperimentConfig.model_validate(payload)

        assert config.training.algorithm == "heuristic"
        assert config.heuristic.type == "dijkstra"
