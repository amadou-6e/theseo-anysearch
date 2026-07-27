"""Unit tests for experiment model and loader behavior."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
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
