"""Unit tests for `_TensorBoardRunWriter`'s degraded/disabled observability.

Covers issue #120: TensorBoard being unavailable must be distinguished from a
real programming error in the integration.

- A genuine ``ImportError``/``ModuleNotFoundError`` (torch/tensorboard not
  installed) is caught, recorded as a disabled reason, and does not raise —
  training continues with TensorBoard disabled.
- Any other exception raised while importing/constructing the writer is a
  real bug and must propagate instead of being silently treated as "not
  installed".
- MLflow tracking failures analogously must leave a durable record in the
  run's persisted artifacts, not just an ephemeral warning.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from theseo_anysearch.rllib.trainer.reporting.tensorboard import _TensorBoardRunWriter


class TestTensorBoardImportMissing:
    """A genuine ImportError is treated as the benign 'not installed' case."""

    def test_module_not_found_disables_without_raising(self, tmp_path: Path):
        with patch.dict(sys.modules, {"torch.utils.tensorboard": None}):
            writer = _TensorBoardRunWriter(tmp_path)  # must not raise

        assert writer.enabled is False
        assert writer.disabled_reason is not None
        assert "ModuleNotFoundError" in writer.disabled_reason

    def test_training_continues_when_disabled(self, tmp_path: Path):
        """log_iteration/log_scalars/close must all be safe no-ops when disabled."""
        from theseo_anysearch.rllib.trainer.results import TrainResult

        with patch.dict(sys.modules, {"torch.utils.tensorboard": None}):
            writer = _TensorBoardRunWriter(tmp_path)

        result = TrainResult(
            iteration=1,
            episode_reward_mean=1.0,
            episode_len_mean=2.0,
            episodes_total=3,
            elapsed_s=0.1,
        )
        writer.log_iteration(result)  # must not raise
        writer.log_scalars(1, {"custom/metric": 1.0})  # must not raise
        writer.close()  # must not raise

    def test_disabled_reason_recorded_in_run_artifacts(self, tmp_path: Path):
        """The disabled reason must be persisted to disk, not just logged."""
        with patch.dict(sys.modules, {"torch.utils.tensorboard": None}):
            _TensorBoardRunWriter(tmp_path)

        status_path = tmp_path.joinpath("tensorboard_status.json")
        assert status_path.exists()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["enabled"] is False
        assert status["disabled_reason"]


class TestTensorBoardRealErrorPropagates:
    """A non-ImportError failure during setup is a real bug and must surface."""

    def test_non_import_error_is_not_swallowed(self, tmp_path: Path):
        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch.utils.tensorboard":
                raise TypeError("simulated code defect, not a missing dependency")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            with pytest.raises(TypeError, match="simulated code defect"):
                _TensorBoardRunWriter(tmp_path)

    def test_attribute_error_is_not_swallowed(self, tmp_path: Path):
        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch.utils.tensorboard":
                raise AttributeError("simulated attribute error, not missing-dependency")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            with pytest.raises(AttributeError, match="simulated attribute error"):
                _TensorBoardRunWriter(tmp_path)

    def test_no_status_file_written_when_error_propagates(self, tmp_path: Path):
        """A real bug must not be mislabeled as a recorded 'disabled' state."""
        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch.utils.tensorboard":
                raise TypeError("boom")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            with pytest.raises(TypeError):
                _TensorBoardRunWriter(tmp_path)

        assert not tmp_path.joinpath("tensorboard_status.json").exists()
