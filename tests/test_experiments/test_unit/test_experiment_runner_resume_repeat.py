"""Unit tests for experiment runner resume and repeat flows."""

from __future__ import annotations

import json

import pytest

from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.experiments.runner import ExperimentRunner, RunInfo

from ._support import patch_build


class TestExperimentRunnerResume:
    """Verify resume behavior for existing runs."""

    def _do_run(self, config, runner_mod):
        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            info = ExperimentRunner(config).run()
        finally:
            runner_mod._build_trainer = original
        return info

    def test_resume_completes(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        info = self._do_run(experiment_config, runner_mod)

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            resumed = ExperimentRunner(experiment_config).resume(info.run_id)
        finally:
            runner_mod._build_trainer = original

        assert resumed.status == "COMPLETED"

    def test_resume_preserves_run_id(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        info = self._do_run(experiment_config, runner_mod)

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            resumed = ExperimentRunner(experiment_config).resume(info.run_id)
        finally:
            runner_mod._build_trainer = original

        assert resumed.run_id == info.run_id

    def test_resume_unknown_run_raises(self, experiment_config: ExperimentConfig):
        runner = ExperimentRunner(experiment_config)
        with pytest.raises(FileNotFoundError):
            runner.resume("deadbeef")

    def test_resume_no_checkpoint_raises(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        run_id = "nocktst1"
        run_dir = experiment_config.run_output_dir.joinpath(run_id)
        run_dir.mkdir(parents=True)
        info = RunInfo(
            run_id=run_id,
            experiment_name=experiment_config.experiment.name,
            start_time="2026-01-01T00:00:00+00:00",
            status="COMPLETED",
        )
        run_dir.joinpath("run.json").write_text(
            json.dumps(info.model_dump()),
            encoding="utf-8",
        )

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            with pytest.raises(FileNotFoundError):
                ExperimentRunner(experiment_config).resume(run_id)
        finally:
            runner_mod._build_trainer = original


class TestExperimentRunnerRepeat:
    """Verify repeat behavior for existing runs."""

    def _do_run_with_yaml(self, config, yaml_path, runner_mod):
        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            info = ExperimentRunner(config, yaml_path).run()
        finally:
            runner_mod._build_trainer = original
        return info

    def test_repeat_returns_new_run_id(
        self,
        experiment_config: ExperimentConfig,
        single_yaml,
    ):
        import theseo_anysearch.experiments.runner as runner_mod

        info1 = self._do_run_with_yaml(experiment_config, single_yaml, runner_mod)

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            info2 = ExperimentRunner(experiment_config).repeat(info1.run_id)
        finally:
            runner_mod._build_trainer = original

        assert info2.run_id != info1.run_id

    def test_repeat_status_completed(
        self,
        experiment_config: ExperimentConfig,
        single_yaml,
    ):
        import theseo_anysearch.experiments.runner as runner_mod

        info1 = self._do_run_with_yaml(experiment_config, single_yaml, runner_mod)

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            info2 = ExperimentRunner(experiment_config).repeat(info1.run_id)
        finally:
            runner_mod._build_trainer = original

        assert info2.status == "COMPLETED"

    def test_repeat_unknown_run_raises(self, experiment_config: ExperimentConfig):
        with pytest.raises(FileNotFoundError):
            ExperimentRunner(experiment_config).repeat("deadbeef")

    def test_repeat_preserves_experiment_name(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            info1 = ExperimentRunner(experiment_config).run()
            info2 = ExperimentRunner(experiment_config).repeat(info1.run_id)
        finally:
            runner_mod._build_trainer = original

        assert info2.experiment_name == info1.experiment_name
