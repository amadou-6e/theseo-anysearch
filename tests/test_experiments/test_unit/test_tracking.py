"""
Unit tests for MLflowTracker and _flatten_config.

All mlflow calls are mocked — no live MLflow server required.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from theseo_anysearch.experiments.models import ExperimentConfig, MLflowConfig
from theseo_anysearch.experiments.tracking import MLflowTracker, _flatten_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlflow_config(**kwargs) -> MLflowConfig:
    return MLflowConfig(**kwargs)


def _make_fake_run(run_id: str = "abc123") -> MagicMock:
    fake = MagicMock()
    fake.info.run_id = run_id
    return fake


def _make_experiment_config(tmp_path: Path) -> ExperimentConfig:
    import yaml
    raw = textwrap.dedent(f"""\
        experiment:
          name: test-exp
          output_dir: {tmp_path}
          seed: 7
        env:
          stl_path: /tmp/test.stl
          agent_count: 1
          max_steps: 50
          seed: 0
        training:
          algorithm: ppo
          iterations: 5
          checkpoint_interval: 1
          output_dir: {tmp_path}
          video_every: 10
        algorithm_config:
          lr: 0.001
          gamma: 0.99
          train_batch_size: 512
        model_config:
          hidden_sizes: [128, 64]
          activation: relu
    """)
    data = yaml.safe_load(raw)
    return ExperimentConfig.model_validate(data)


# ---------------------------------------------------------------------------
# No-op mode (mlflow_config = None)
# ---------------------------------------------------------------------------

class TestMLflowTrackerNoOp:
    """Tests MLflowTrackerNoOp."""
    def _tracker(self) -> MLflowTracker:
        return MLflowTracker(None, "my-exp")

    def test_start_run_returns_none(self):
        t = self._tracker()
        assert t.start_run("run-1") is None

    def test_start_child_run_returns_none(self):
        t = self._tracker()
        assert t.start_child_run("trial-1") is None

    def test_end_run_is_noop(self):
        t = self._tracker()
        t.end_run("FINISHED")  # must not raise

    def test_end_child_run_is_noop(self):
        t = self._tracker()
        t.end_child_run("FAILED")  # must not raise

    def test_log_params_is_noop(self):
        t = self._tracker()
        t.log_params({"lr": 0.001, "gamma": 0.99})  # must not raise

    def test_log_metrics_is_noop(self):
        t = self._tracker()
        t.log_metrics({"episode_reward_mean": 0.5}, step=10)  # must not raise

    def test_log_artifact_is_noop(self):
        t = self._tracker()
        t.log_artifact("/tmp/some.pb")  # must not raise

    def test_set_tag_is_noop(self):
        t = self._tracker()
        t.set_tag("key", "value")  # must not raise

    def test_run_url_is_none(self):
        t = self._tracker()
        assert t.run_url is None


# ---------------------------------------------------------------------------
# Enabled mode with mocked mlflow
# ---------------------------------------------------------------------------

class TestMLflowTrackerWithMock:
    """All mlflow calls are intercepted via patch; no server required."""

    def _make_tracker(self, patches: dict) -> MLflowTracker:
        cfg = _make_mlflow_config(tracking_uri="http://localhost:5000")
        return MLflowTracker(cfg, "my-exp")

    def test_start_run_returns_run_id(self):
        fake_run = _make_fake_run("run-xyz")
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run", return_value=fake_run) as mock_start:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            run_id = t.start_run("run-name", tags={"k": "v"})
        assert run_id == "run-xyz"
        mock_start.assert_called_once_with(run_name="run-name", tags={"k": "v"})

    def test_start_child_run_uses_nested_flag(self):
        fake_run = _make_fake_run("child-id")
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run", return_value=_make_fake_run()), \
             patch("mlflow.start_run", return_value=fake_run) as mock_start:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            # First start_run sets up parent; second should be nested
        with patch("mlflow.start_run", return_value=fake_run) as mock_nested:
            t._enabled = True  # keep enabled after init
            child_id = t.start_child_run("trial-1")
        mock_nested.assert_called_once_with(
            run_name="trial-1", tags=None, nested=True
        )
        assert child_id == "child-id"

    def test_end_run_calls_mlflow_end_run(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.end_run") as mock_end:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            t.end_run("FAILED")
        mock_end.assert_called_once_with(status="FAILED")

    def test_end_child_run_delegates_to_end_run(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.end_run") as mock_end:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            t.end_child_run("FINISHED")
        mock_end.assert_called_once_with(status="FINISHED")

    def test_log_params_stringifies_values(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.log_params") as mock_log:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            t.log_params({"lr": 0.001, "gamma": 0.99, "name": "ppo"})
        args = mock_log.call_args[0][0]
        assert args == {"lr": "0.001", "gamma": "0.99", "name": "ppo"}

    def test_log_params_truncates_long_values(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.log_params") as mock_log:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            t.log_params({"k": "x" * 600})
        args = mock_log.call_args[0][0]
        assert len(args["k"]) == 500

    def test_log_metrics_passes_step(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.log_metrics") as mock_log:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            t.log_metrics({"episode_reward_mean": 0.75}, step=42)
        mock_log.assert_called_once_with({"episode_reward_mean": 0.75}, step=42)

    def test_set_tag_calls_mlflow_set_tag(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.set_tag") as mock_tag:
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            t.set_tag("best_trial_id", "abc")
        mock_tag.assert_called_once_with("best_trial_id", "abc")

    def test_run_url_none_for_local_store(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlruns"), \
             patch("mlflow.set_experiment"):
            t = MLflowTracker(MLflowConfig(), "my-exp")
        t._run_id = "some-id"
        assert t.run_url is None

    def test_run_url_constructed_for_http_store(self):
        fake_run_obj = MagicMock()
        fake_run_obj.info.experiment_id = "1"
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.tracking.MlflowClient") as MockClient:
            MockClient.return_value.get_run.return_value = fake_run_obj
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            t._run_id = "xyz"
            url = t.run_url
        assert url == "http://localhost:5000/#/experiments/1/runs/xyz"

    def test_mlflow_error_does_not_raise(self):
        """All mlflow failures are swallowed as warnings, never propagated."""
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.log_metrics", side_effect=RuntimeError("conn refused")):
            t = MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"), "my-exp"
            )
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t.log_metrics({"x": 1.0}, step=1)
            assert len(w) == 1
            assert "conn refused" in str(w[0].message)

    def test_experiment_name_override(self):
        """MLflowConfig.experiment_name takes priority over the default."""
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment") as mock_set_exp:
            MLflowTracker(
                _make_mlflow_config(
                    tracking_uri="http://localhost:5000",
                    experiment_name="custom-name",
                ),
                "default-name",
            )
        mock_set_exp.assert_called_once_with("custom-name")

    def test_experiment_name_falls_back_to_default(self):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"), \
             patch("mlflow.set_experiment") as mock_set_exp:
            MLflowTracker(
                _make_mlflow_config(tracking_uri="http://localhost:5000"),
                "default-name",
            )
        mock_set_exp.assert_called_once_with("default-name")


# ---------------------------------------------------------------------------
# _flatten_config
# ---------------------------------------------------------------------------

class TestFlattenConfig:
    """Tests FlattenConfig."""
    def test_dotted_keys(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert "env.agent_count" in flat
        assert "training.algorithm" in flat
        assert "algorithm_config.lr" in flat

    def test_excludes_mlflow(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert not any(k.startswith("mlflow") for k in flat)

    def test_excludes_anyscale(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert not any(k.startswith("anyscale") for k in flat)

    def test_excludes_renders(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert not any(k.startswith("renders") for k in flat)

    def test_excludes_training_output_dir(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert "training.output_dir" not in flat

    def test_excludes_experiment_output_dir(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert "experiment.output_dir" not in flat

    def test_list_json_encoded(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        # model_config.hidden_sizes is [128, 64]
        assert "model_config.hidden_sizes" in flat
        parsed = json.loads(flat["model_config.hidden_sizes"])
        assert parsed == [128, 64]

    def test_scalar_values_preserved(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert flat["env.agent_count"] == 1
        assert flat["algorithm_config.lr"] == pytest.approx(0.001)

    def test_none_values_excluded(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        # tune_config is None — should not appear
        assert not any(k.startswith("tune_config") for k in flat)

    def test_experiment_name_included(self, tmp_path):
        config = _make_experiment_config(tmp_path)
        flat = _flatten_config(config)
        assert flat.get("experiment.name") == "test-exp"
