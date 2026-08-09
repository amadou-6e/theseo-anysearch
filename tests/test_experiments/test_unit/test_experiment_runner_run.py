"""Unit tests for experiment runner execution paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from theseo_anysearch.experiments.models import ExperimentConfig, HeuristicConfig
from theseo_anysearch.experiments.runner import ExperimentRunner, RunInfo

from ._support import patch_build


class TestExperimentRunnerRun:
    """Verify normal runner execution behavior."""

    def test_run_creates_run_json(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            info = runner.run()
        finally:
            runner_mod._build_trainer = original

        assert info.status == "COMPLETED"
        assert len(info.run_id) == 8

    def test_run_output_dir_created(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            info = runner.run()
        finally:
            runner_mod._build_trainer = original

        run_dir = experiment_config.run_output_dir.joinpath(info.run_id)
        assert run_dir.is_dir()

    def test_run_writes_experiment_yaml(
        self,
        experiment_config: ExperimentConfig,
        single_yaml: Path,
    ):
        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config, single_yaml)
            info = runner.run()
        finally:
            runner_mod._build_trainer = original

        run_dir = experiment_config.run_output_dir.joinpath(info.run_id)
        assert run_dir.joinpath("experiment.yaml").exists()

    def test_run_checkpoints_written(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            info = runner.run()
        finally:
            runner_mod._build_trainer = original

        assert info.checkpoint_iterations == [1, 2, 3]

    def test_staged_run_completes_all_stages(
        self,
        experiment_config: ExperimentConfig,
    ):
        import theseo_anysearch.experiments.runner as runner_mod

        payload = experiment_config.model_dump(by_alias=True, mode="python")
        payload["staging"] = {
            "replay_transition": "clear",
            "stages": [
                {
                    "name": "one-step",
                    "iterations": 1,
                    "env": {"max_steps": 1, "trail_mode": False},
                },
                {
                    "name": "with-trails",
                    "iterations": 2,
                    "env": {"max_steps": 50, "trail_mode": True},
                },
            ],
        }
        config = ExperimentConfig.model_validate(payload)
        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            info = ExperimentRunner(config).run()
        finally:
            runner_mod._build_trainer = original

        run_dir = config.run_output_dir.joinpath(info.run_id)
        state = json.loads(
            run_dir.joinpath("staging_state.json").read_text(encoding="utf-8")
        )
        assert info.status == "COMPLETED"
        assert state["completed_stages"] == ["one-step", "with-trails"]
        assert state["completed_iterations"] == 3
        assert info.checkpoint_iterations == [1, 2, 3]

    def test_run_collects_enabled_heuristic_reference(
        self,
        experiment_config: ExperimentConfig,
    ):
        import theseo_anysearch.experiments.runner as runner_mod

        config = experiment_config.model_copy(
            update={
                "heuristic": HeuristicConfig(
                    enabled=True,
                    type="weighted_astar",
                    weight=2.0,
                )
            }
        )
        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        episode = MagicMock(name="heuristic_episode")
        try:
            with (
                patch(
                    "theseo_anysearch.experiments.trajectory.collect_heuristic_episode",
                    return_value=episode,
                ) as collect,
                patch(
                    "theseo_anysearch.experiments.trajectory.write_heuristic_trajectory",
                    return_value="trajectories/heuristic_weighted_astar.json",
                ) as write,
            ):
                info = ExperimentRunner(config).run()
        finally:
            runner_mod._build_trainer = original

        assert info.status == "COMPLETED"
        collect.assert_called_once()
        assert collect.call_args.args[1] == "weighted_astar"
        assert collect.call_args.kwargs["weight"] == pytest.approx(2.0)
        write.assert_called_once()
    def test_standalone_heuristic_skips_trainer_build(
        self,
        experiment_config: ExperimentConfig,
    ):
        import theseo_anysearch.experiments.runner as runner_mod

        config = experiment_config.model_copy(
            update={
                "training": experiment_config.training.model_copy(
                    update={"algorithm": "heuristic"}
                ),
                "heuristic": HeuristicConfig(enabled=True, type="dijkstra"),
            }
        )
        episode = MagicMock(name="heuristic_episode")
        with (
            patch.object(runner_mod, "_build_trainer") as build_trainer,
            patch(
                "theseo_anysearch.experiments.trajectory.collect_heuristic_episode",
                return_value=episode,
            ) as collect,
            patch(
                "theseo_anysearch.experiments.trajectory.write_heuristic_trajectory",
                return_value="trajectories/heuristic_dijkstra.json",
            ) as write,
        ):
            info = ExperimentRunner(config).run()

        assert info.status == "COMPLETED"
        build_trainer.assert_not_called()
        collect.assert_called_once()
        assert collect.call_args.args[1] == "dijkstra"
        write.assert_called_once()

class TestExperimentRunnerFailure:
    """Verify runner failure bookkeeping."""

    def test_failed_run_status_is_failed(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer

        class _BrokenAlgo:
            """Raise during train to exercise failure handling."""

            def train(self):
                raise RuntimeError("algo exploded")

            def save(self, path: str) -> str:
                checkpoint_dir = Path(path)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                return path

            def restore(self, path: str) -> None:
                return None

        class _BrokenPPO(PPOTrainer):
            """Fake PPO wrapper that returns the broken algorithm."""

            def _build_algorithm(self):
                return _BrokenAlgo()

        original = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            settings = cfg.to_settings()
            settings = settings.model_copy(
                update={"training": settings.training.model_copy(update={"output_dir": output_dir})}
            )
            return _BrokenPPO(settings)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            with pytest.raises(RuntimeError, match="algo exploded"):
                runner.run()
        finally:
            runner_mod._build_trainer = original

        run_dirs = list(experiment_config.run_output_dir.iterdir())
        assert len(run_dirs) == 1
        info = RunInfo.model_validate(
            json.loads(run_dirs[0].joinpath("run.json").read_text(encoding="utf-8"))
        )
        assert info.status == "FAILED"

    def test_failed_run_end_time_set(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer

        class _BrokenAlgo:
            """Raise during train to ensure end time is written."""

            def train(self):
                raise ValueError("boom")

            def save(self, path: str) -> str:
                checkpoint_dir = Path(path)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                return path

            def restore(self, path: str) -> None:
                return None

        class _BrokenPPO(PPOTrainer):
            """Fake PPO wrapper that returns the broken algorithm."""

            def _build_algorithm(self):
                return _BrokenAlgo()

        original = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            settings = cfg.to_settings()
            settings = settings.model_copy(
                update={"training": settings.training.model_copy(update={"output_dir": output_dir})}
            )
            return _BrokenPPO(settings)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            with pytest.raises(ValueError):
                runner.run()
        finally:
            runner_mod._build_trainer = original

        run_dir = next(experiment_config.run_output_dir.iterdir())
        info = RunInfo.model_validate(
            json.loads(run_dir.joinpath("run.json").read_text(encoding="utf-8"))
        )
        assert info.end_time is not None
