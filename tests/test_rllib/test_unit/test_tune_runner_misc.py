"""Unit tests for tune runner naming, GPU detection, and overview output."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTrialDirname:
    """Verify short trial directory naming for Windows path safety."""

    def test_returns_trial_prefix_plus_id(self):
        from theseo_anysearch.experiments.tune_runner import _trial_dirname

        mock_trial = MagicMock()
        mock_trial.trial_id = "abc123"
        assert _trial_dirname(mock_trial) == "trial_abc123"

    def test_does_not_embed_param_values(self):
        from theseo_anysearch.experiments.tune_runner import _trial_dirname

        mock_trial = MagicMock()
        mock_trial.trial_id = "xyz"
        mock_trial.config = {
            "lr": 0.000123456789,
            "train_batch_size": 8192,
            "kl_coeff": 0.9999,
        }
        result = _trial_dirname(mock_trial)
        assert "0.000123" not in result
        assert "8192" not in result

    def test_length_safe_for_windows_max_path(self):
        from theseo_anysearch.experiments.tune_runner import _trial_dirname

        mock_trial = MagicMock()
        mock_trial.trial_id = "a" * 20
        assert len(_trial_dirname(mock_trial)) < 50


class TestDetectNumGpusOverride:
    """Verify explicit GPU override behavior in trainer setup."""

    def test_override_returns_fractional(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        assert _detect_num_gpus(require_gpu=True, num_gpus=0.5) == pytest.approx(0.5)

    def test_override_zero_skips_torch(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        with patch("torch.cuda.device_count", return_value=0):
            assert _detect_num_gpus(require_gpu=False, num_gpus=0.0) == pytest.approx(0.0)

    def test_override_none_falls_through_to_torch(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        with patch("torch.cuda.device_count", return_value=2):
            assert _detect_num_gpus(require_gpu=False, num_gpus=None) == 2

    def test_override_one_does_not_raise_without_cuda(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        assert _detect_num_gpus(require_gpu=True, num_gpus=1.0) == pytest.approx(1.0)

    def test_require_gpu_without_override_raises_if_no_cuda(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        with patch("torch.cuda.device_count", return_value=0):
            with pytest.raises(AssertionError, match="require_gpu"):
                _detect_num_gpus(require_gpu=True, num_gpus=None)


class TestPrintSweepOverview:
    """Verify the sweep banner prints human-readable sampler descriptions."""

    def _capture_overview(self, search_space: dict) -> str:
        from theseo_anysearch.experiments.tune_runner import _print_sweep_overview

        buffer = StringIO()
        with patch("sys.stdout", buffer):
            _print_sweep_overview(
                config_path=None,
                run_tag="test",
                algorithm="multi_agent_voxel_ppo",
                scheduler="asha",
                num_samples=10,
                max_concurrent=1,
                iterations=100,
                num_env_runners=4,
                require_gpu=False,
                search_space=search_space,
                sweep_dir=Path("/tmp/out"),
                mlflow_tracking_uri="",
            )
        return buffer.getvalue()

    def test_no_object_repr_in_output(self):
        import ray.tune as ray_tune

        output = self._capture_overview(
            {
                "lr": ray_tune.loguniform(1e-5, 1e-2),
                "batch": ray_tune.choice([256, 512]),
            }
        )
        assert "object at 0x" not in output

    def test_categorical_shows_choices(self):
        import ray.tune as ray_tune

        output = self._capture_overview({"batch": ray_tune.choice([256, 512, 1024])})
        assert "256" in output
        assert "512" in output
        assert "1024" in output

    def test_param_name_appears(self):
        import ray.tune as ray_tune

        output = self._capture_overview({"learning_rate": ray_tune.loguniform(1e-5, 1e-2)})
        assert "learning_rate" in output

    def test_no_gpu_shows_no(self):
        output = self._capture_overview({})
        assert "GPU" in output
        assert "no" in output

    def test_require_gpu_shows_yes(self):
        from theseo_anysearch.experiments.tune_runner import _print_sweep_overview

        buffer = StringIO()
        with patch("sys.stdout", buffer):
            _print_sweep_overview(
                config_path=None,
                run_tag="test",
                algorithm="ppo",
                scheduler="asha",
                num_samples=5,
                max_concurrent=1,
                iterations=50,
                num_env_runners=4,
                require_gpu=True,
                search_space={},
                sweep_dir=Path("/tmp/out"),
                mlflow_tracking_uri="",
            )
        output = buffer.getvalue()
        assert "GPU" in output
        assert "yes" in output
