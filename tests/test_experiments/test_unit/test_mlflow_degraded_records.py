"""Unit tests for issue #120: degraded MLflow tracking must be recorded in
the run's persisted artifacts, not just logged as a transient warning.

A run whose MLflow tracking silently stops working partway through training
must be distinguishable, after the fact, from a run that never configured
MLflow at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from theseo_anysearch.experiments.models import ExperimentConfig, MLflowConfig
from theseo_anysearch.experiments.runner import ExperimentRunner, RunInfo

from ._support import patch_build


def _make_fake_run(run_id: str = "run-id-1") -> MagicMock:
    fake = MagicMock()
    fake.info.run_id = run_id
    return fake


class TestMLflowDegradedRecordedInRunArtifacts:
    """Failures inside MLflow calls must survive into run.json after the run."""

    def test_log_metrics_failure_marks_run_degraded(
        self,
        experiment_config: ExperimentConfig,
        tmp_path: Path,
    ):
        config = experiment_config.model_copy(
            update={
                "mlflow": MLflowConfig(
                    tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"
                )
            }
        )

        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            with (
                patch("mlflow.set_tracking_uri"),
                patch("mlflow.set_experiment"),
                patch("mlflow.start_run", return_value=_make_fake_run()),
                patch("mlflow.log_params"),
                patch("mlflow.log_artifact"),
                patch(
                    "mlflow.log_metrics",
                    side_effect=RuntimeError("mlflow backend unreachable"),
                ),
                patch("mlflow.end_run"),
            ):
                info = ExperimentRunner(config).run()
        finally:
            runner_mod._build_trainer = original

        # Training itself must not fail just because tracking degraded.
        assert info.status == "COMPLETED"

        # The degraded state and its reason must be visible on the in-memory
        # result...
        assert info.mlflow_degraded is True
        assert "mlflow backend unreachable" in (info.mlflow_degraded_reason or "")

        # ...and persisted to the run's run.json artifact, so a user
        # inspecting the run directory afterward can tell tracking degraded
        # and why, rather than seeing a run that looks identical to one that
        # never configured MLflow at all.
        run_dir = config.run_output_dir.joinpath(info.run_id)
        persisted = RunInfo.model_validate(
            json.loads(run_dir.joinpath("run.json").read_text(encoding="utf-8"))
        )
        assert persisted.mlflow_degraded is True
        assert "mlflow backend unreachable" in (persisted.mlflow_degraded_reason or "")

    def test_healthy_run_is_not_marked_degraded(
        self,
        experiment_config: ExperimentConfig,
        tmp_path: Path,
    ):
        """A run that never configures MLflow must not look degraded."""
        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, original = patch_build(runner_mod)
        runner_mod._build_trainer = fake_build
        try:
            info = ExperimentRunner(experiment_config).run()
        finally:
            runner_mod._build_trainer = original

        assert info.status == "COMPLETED"
        assert info.mlflow_degraded is False
        assert info.mlflow_degraded_reason is None

        run_dir = experiment_config.run_output_dir.joinpath(info.run_id)
        persisted = RunInfo.model_validate(
            json.loads(run_dir.joinpath("run.json").read_text(encoding="utf-8"))
        )
        assert persisted.mlflow_degraded is False
        assert persisted.mlflow_degraded_reason is None
