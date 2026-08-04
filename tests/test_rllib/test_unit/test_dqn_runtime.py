"""Tests for AnySearch DQN driver-overhead extensions."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from ray.rllib.algorithms.dqn import DQN as RllibDQN
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.core import ALL_MODULES

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


def _algorithm_with_sync_interval(interval: int) -> tuple[AnySearchDQN, Mock, Mock]:
    algorithm = object.__new__(AnySearchDQN)
    sync_weights = Mock()
    algorithm.env_runner_group = SimpleNamespace(sync_weights=sync_weights)
    learner_update = Mock(
        return_value=[{"default_policy": {}, ALL_MODULES: {}}]
    )
    algorithm.learner_group = SimpleNamespace(update=learner_update)
    config = SimpleNamespace(
        learner_config_dict={"weight_sync_interval": interval}
    )

    with patch.object(RllibDQN, "setup", return_value=None):
        algorithm.setup(config)
    return algorithm, learner_update, sync_weights


def test_weight_sync_interval_counts_learner_updates() -> None:
    """An interval of four broadcasts after every fourth learner update."""
    algorithm, learner_update, sync_weights = _algorithm_with_sync_interval(4)

    for _ in range(8):
        algorithm.learner_group.update(batch="replay")

    assert learner_update.call_count == 8
    assert sync_weights.call_count == 2
    assert sync_weights.call_args.kwargs["policies"] == {"default_policy"}


def test_weight_sync_interval_one_preserves_default_behavior() -> None:
    """The default interval broadcasts after every learner update."""
    algorithm, _, sync_weights = _algorithm_with_sync_interval(1)

    for _ in range(3):
        algorithm.learner_group.update()

    assert sync_weights.call_count == 3


def test_outer_dqn_sync_is_suppressed() -> None:
    """RLlib's outer-call synchronization does not duplicate update syncs."""
    algorithm, _, sync_weights = _algorithm_with_sync_interval(1)

    algorithm.env_runner_group.sync_weights()

    sync_weights.assert_not_called()
