"""Unit tests for PPO rollout worker configuration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_configure_rllib_env_runners_propagates_vectorization_and_resources():
    from theseo_anysearch.rllib.trainer.ppo import _configure_rllib_env_runners

    rllib_config = MagicMock()
    configured = MagicMock()
    rllib_config.env_runners.return_value = configured
    training = SimpleNamespace(
        num_env_runners=2,
        num_envs_per_env_runner=4,
        num_gpus_per_env_runner=0.0,
    )

    result = _configure_rllib_env_runners(rllib_config, training)

    assert result is configured
    rllib_config.env_runners.assert_called_once_with(
        num_env_runners=2,
        num_envs_per_env_runner=4,
        num_gpus_per_env_runner=0.0,
    )


def test_installed_rllib_config_retains_vectorization_and_cpu_inference():
    pytest.importorskip("ray")
    from ray.rllib.algorithms.ppo import PPOConfig
    from theseo_anysearch.rllib.trainer.ppo import _configure_rllib_env_runners

    training = SimpleNamespace(
        num_env_runners=2,
        num_envs_per_env_runner=4,
        num_gpus_per_env_runner=0.0,
    )

    configured = _configure_rllib_env_runners(PPOConfig(), training)

    assert configured.num_env_runners == 2
    assert configured.num_envs_per_env_runner == 4
    assert configured.num_gpus_per_env_runner == 0.0