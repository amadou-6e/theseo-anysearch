"""Integration test covering a real PPO trainer run against the real environment stack."""

from __future__ import annotations

from pathlib import Path

import pytest

from theseo_anysearch.models import (
    AnyscaleConfig,
    EnvConfig,
    EvaluationConfig,
    GeometryConfig,
    Settings,
    TrainingConfig,
)
from theseo_anysearch.rllib.algorithms.models import PPOConfig
from theseo_anysearch.rllib.models.models import VoxelEncoderConfig
from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer


pytestmark = pytest.mark.ray


@pytest.mark.timeout(60)
def test_ppo_trainer_runs_real_ray_with_real_env(tmp_path: Path) -> None:
    """Validate ppo trainer runs real ray with real env."""
    settings = Settings(
        env=EnvConfig(
            geometry=GeometryConfig(boxes=[]),
            agent_count=1,
            max_steps=8,
            seed=0,
        ),
        training=TrainingConfig(
            algorithm="ppo",
            model="voxel_encoder",
            runner="local",
            iterations=2,
            checkpoint_interval=1,
            output_dir=tmp_path.joinpath("out"),
            video_every=999,
        ),
        anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
        algorithm_config=PPOConfig(
            lr=1.0e-3,
            gamma=0.99,
            train_batch_size=32,
            minibatch_size=16,
            clip_param=0.2,
            num_sgd_iter=2,
            lambda_=0.95,
            kl_coeff=0.2,
        ),
        model_config=VoxelEncoderConfig(hidden_sizes=[32], activation="relu"),
    )

    results = PPOTrainer(settings).train()

    assert len(results) == 2
    assert results[-1].iteration == 2
    assert tmp_path.joinpath("out", "ray_runtime.json").is_file()
    assert tmp_path.joinpath("out", "checkpoints", "latest.json").is_file()


@pytest.mark.timeout(60)
def test_ppo_uses_rllib_parallel_custom_evaluation(tmp_path: Path) -> None:
    """Validate RLlib overlaps and returns AnySearch evaluation episodes."""
    settings = Settings(
        env=EnvConfig(
            geometry=GeometryConfig(boxes=[]),
            agent_count=1,
            max_steps=8,
            seed=0,
        ),
        training=TrainingConfig(
            algorithm="ppo",
            model="voxel_encoder",
            runner="local",
            iterations=1,
            checkpoint_interval=1,
            output_dir=tmp_path.joinpath("parallel-out"),
            video_every=999,
        ),
        evaluation=EvaluationConfig(
            episodes=2,
            num_env_runners=1,
            num_envs_per_env_runner=2,
            parallel_to_training=True,
        ),
        anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
        algorithm_config=PPOConfig(
            lr=1.0e-3,
            gamma=0.99,
            train_batch_size=32,
            minibatch_size=16,
            clip_param=0.2,
            num_sgd_iter=1,
            lambda_=0.95,
            kl_coeff=0.2,
        ),
        model_config=VoxelEncoderConfig(hidden_sizes=[32], activation="relu"),
    )

    results = PPOTrainer(settings).train()

    assert len(results) == 1
    assert results[0].evaluation_episodes == 2
