"""Unit coverage for Tune reporting, checkpointing, pruning, and budgets."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from theseo_anysearch.experiments.models import TuneConfig
from theseo_anysearch.experiments.tune_runner import (
    _persist_trial_outcomes,
    _ray_checkpoint,
    _restore_reported_checkpoint,
    _selected_metric,
    _trial_resource_metrics,
    _tune_stop_criteria,
)


def test_success_rate_is_the_selected_optimization_metric() -> None:
    payload = {
        "episode_reward_mean": 8.0,
        "evaluation_success_rate": 0.75,
    }

    assert _selected_metric(
        payload,
        "evaluation_success_rate",
        fallback=8.0,
        mode="max",
    ) == pytest.approx(0.75)


def test_non_finite_selected_metric_is_sanitized_by_mode() -> None:
    assert _selected_metric(
        {"evaluation_success_rate": float("nan")},
        "evaluation_success_rate",
        fallback=0.0,
        mode="max",
    ) == -1e9


def test_stop_criteria_cover_all_explicit_budgets() -> None:
    config = TuneConfig(
        max_environment_steps=12_000,
        max_wall_time_s=90.0,
        target_success_rate=0.8,
    )

    assert _tune_stop_criteria(config, max_iterations=40) == {
        "training_iteration": 40,
        "environment_steps_total": 12_000,
        "time_total_s": 90.0,
        "evaluation_success_rate": 0.8,
    }


def test_project_checkpoint_is_attached_as_ray_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path.joinpath("checkpoints", "iter_000003")
    checkpoint_dir.mkdir(parents=True)

    with patch(
        "ray.train.Checkpoint.from_directory",
        return_value="ray-checkpoint",
    ) as from_directory:
        checkpoint = _ray_checkpoint(checkpoint_dir)

    assert checkpoint == "ray-checkpoint"
    from_directory.assert_called_once_with(str(checkpoint_dir))


class _IncomingCheckpoint:
    __module__ = "ray.fake"

    @contextmanager
    def as_directory(self):
        yield "restored-checkpoint"


def test_resume_restores_trainer_from_reported_checkpoint() -> None:
    trainer = MagicMock()
    tune_api = MagicMock()
    tune_api.get_checkpoint.return_value = _IncomingCheckpoint()

    assert _restore_reported_checkpoint(trainer, tune_api) is True
    trainer.restore.assert_called_once_with(Path("restored-checkpoint"))


def test_pruned_trial_status_is_persisted_below_iteration_budget(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path.joinpath("trial-a")
    trial_dir.mkdir()
    result = SimpleNamespace(
        metrics={"trial_id": "trial-a", "training_iteration": 3},
        error=None,
    )

    _persist_trial_outcomes(tmp_path, [result], max_iterations=10)

    status = json.loads(
        trial_dir.joinpath("tune_status.json").read_text(encoding="utf-8")
    )
    assert status == {"status": "PRUNED", "training_iteration": 3}


def test_failed_trial_status_is_persisted(tmp_path: Path) -> None:
    trial_dir = tmp_path.joinpath("trial-b")
    trial_dir.mkdir()
    result = SimpleNamespace(
        metrics={"trial_id": "trial-b", "training_iteration": 2},
        error=RuntimeError("training failed"),
    )

    _persist_trial_outcomes(tmp_path, [result], max_iterations=10)

    status = json.loads(
        trial_dir.joinpath("tune_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "FAILED"


class _Parameter:
    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


def test_resource_report_exposes_compute_and_architecture_cost() -> None:
    policy = SimpleNamespace(
        model=SimpleNamespace(parameters=lambda: [_Parameter(10), _Parameter(6)]),
        observation_space=None,
    )
    trainer = SimpleNamespace(
        _algo=SimpleNamespace(get_policy=lambda: policy),
    )
    settings = SimpleNamespace(
        algorithm_config=SimpleNamespace(
            train_batch_size=1024,
            num_sgd_iter=20,
        ),
        model_cfg=SimpleNamespace(hidden_sizes=[128, 128, 128, 128]),
        training=SimpleNamespace(
            num_env_runners=2,
            num_gpus=0.5,
        ),
    )

    metrics = _trial_resource_metrics(trainer, settings)

    assert metrics["resource/train_batch_size"] == 1024
    assert metrics["resource/num_sgd_iter"] == 20
    assert metrics["resource/model_parameter_count"] == 16
    assert metrics["resource/hidden_layer_count"] == 4
    assert metrics["resource/hidden_layer_width"] == 128


def test_environment_step_budget_metric_supports_new_and_legacy_rllib() -> None:
    from theseo_anysearch.rllib.trainer.base import TrainResult

    current = TrainResult.from_rllib(
        1,
        {"env_runners": {"num_env_steps_sampled_lifetime": 123}},
        0.1,
    )
    legacy = TrainResult.from_rllib(
        1,
        {"timesteps_total": 456},
        0.1,
    )

    assert current.environment_steps_total == 123
    assert legacy.environment_steps_total == 456
    assert current.standard_metrics()["environment_steps_total"] == 123.0


def test_asha_stops_a_deliberately_poor_trial_before_budget() -> None:
    from ray.tune.experiment.trial import Trial
    from ray.tune.schedulers import ASHAScheduler, TrialScheduler

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="evaluation_success_rate",
        mode="max",
        max_t=8,
        grace_period=1,
        reduction_factor=2,
    )
    good = Trial("__fake", trial_id="good")
    poor = Trial("__fake", trial_id="poor")
    scheduler.on_trial_add(None, good)
    scheduler.on_trial_add(None, poor)

    good_decision = scheduler.on_trial_result(
        None,
        good,
        {"training_iteration": 1, "evaluation_success_rate": 1.0},
    )
    poor_decision = scheduler.on_trial_result(
        None,
        poor,
        {"training_iteration": 1, "evaluation_success_rate": 0.0},
    )

    assert good_decision == TrialScheduler.CONTINUE
    assert poor_decision == TrialScheduler.STOP
    assert 1 < 8
