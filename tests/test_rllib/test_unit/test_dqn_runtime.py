"""Tests for AnySearch DQN driver-overhead extensions."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from ray.rllib.algorithms.dqn import DQN as RllibDQN
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner

from theseo_anysearch.rllib.algorithms.dqn_runtime import (
    AnySearchDQN,
    AnySearchDQNTorchLearner,
)


def test_uniform_replay_removes_per_sample_td_errors() -> None:
    """Uniform replay does not transport unused per-sample TD errors."""
    learner = object.__new__(AnySearchDQNTorchLearner)
    learner.config = SimpleNamespace(
        learner_config_dict={"report_td_errors": False}
    )
    learner.metrics = Mock()

    with patch.object(
        DQNTorchLearner,
        "compute_loss_for_module",
        return_value="loss",
    ):
        result = learner.compute_loss_for_module(module_id="default_policy")

    assert result == "loss"
    learner.metrics.delete.assert_called_once_with(
        "default_policy", "td_error", key_error=False
    )


def test_prioritized_replay_retains_per_sample_td_errors() -> None:
    """Prioritized replay retains TD errors for priority updates."""
    learner = object.__new__(AnySearchDQNTorchLearner)
    learner.config = SimpleNamespace(
        learner_config_dict={"report_td_errors": True}
    )
    learner.metrics = Mock()

    with patch.object(
        DQNTorchLearner,
        "compute_loss_for_module",
        return_value="loss",
    ):
        learner.compute_loss_for_module(module_id="default_policy")

    learner.metrics.delete.assert_not_called()


def test_weight_sync_interval_gates_env_runner_broadcasts() -> None:
    """An interval of four broadcasts only every fourth training sync."""
    algorithm = object.__new__(AnySearchDQN)
    sync_weights = Mock()
    algorithm.env_runner_group = SimpleNamespace(sync_weights=sync_weights)
    config = SimpleNamespace(
        learner_config_dict={"weight_sync_interval": 4}
    )

    with patch.object(RllibDQN, "setup", return_value=None):
        algorithm.setup(config)

    for _ in range(8):
        algorithm.env_runner_group.sync_weights(source="learner")

    assert sync_weights.call_count == 2


def test_weight_sync_interval_one_preserves_default_behavior() -> None:
    """The default interval forwards every synchronization request."""
    algorithm = object.__new__(AnySearchDQN)
    sync_weights = Mock()
    algorithm.env_runner_group = SimpleNamespace(sync_weights=sync_weights)
    config = SimpleNamespace(
        learner_config_dict={"weight_sync_interval": 1}
    )

    with patch.object(RllibDQN, "setup", return_value=None):
        algorithm.setup(config)

    for _ in range(3):
        algorithm.env_runner_group.sync_weights()

    assert sync_weights.call_count == 3