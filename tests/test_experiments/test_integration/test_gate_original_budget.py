"""Original PR #217 run budget adapted to the compiled six-gate world."""

from pathlib import Path

import pytest

from theseo_anysearch.environments.action_spaces import ACTION_OFFSETS_26
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.experiments.loader import load_experiment
from theseo_anysearch.experiments.native_extensions import compile_native_extension


EXPERIMENT = Path(
    "usage/experiments/train/large_world_obstacle_pilot/original_budget"
)


def test_original_budget_and_gate_world_are_frozen() -> None:
    config = load_experiment(EXPERIMENT / "experiment.yaml")

    assert config.training.iterations == 400
    assert config.training.checkpoint_interval == 50
    assert config.training.num_env_runners == 3
    assert config.training.num_envs_per_env_runner == 4
    assert config.algorithm_config.train_batch_size == 4096
    assert config.imitation.generation.episodes == 128
    assert config.imitation.pretraining.epochs == 20
    assert config.evaluation.episodes == 10
    assert config.evaluation.waypoint_curriculum.frequency == 5
    assert config.evaluation.waypoint_curriculum.episodes == 3
    assert config.env.geometry.extent == (4096, 2048, 512)
    assert config.env.rewards.custom.name == "segment_countdown_goal"


def test_original_sparse_reward_is_available_to_the_environment() -> None:
    manifest = compile_native_extension(EXPERIMENT)
    env = VoxelEnv({
        "grid_size": 8,
        "max_steps": 4,
        "trail_mode": False,
        "action_mode": "discrete_26",
        "obs_mode": "box",
        "box_radius": 1,
        "waypoint_route": {
            "start": (2, 2, 2),
            "waypoints": [(4, 2, 2)],
        },
        "custom_reward": "segment_countdown_goal",
        "custom_reward_parameters": {
            "additional_budget": 10.0,
            "minimum_reward": 1.0,
        },
        "native_extension_manifest": str(manifest),
        "task": {},
    })
    positive_x = ACTION_OFFSETS_26.index((1, 0, 0))
    try:
        env.reset(seed=42)
        _, first_reward, _, _, _ = env.step(positive_x)
        _, second_reward, terminated, truncated, info = env.step(positive_x)
        assert first_reward == 0.0
        assert second_reward == pytest.approx(10.0)
        assert terminated and not truncated
        assert info["goal_reached"]
    finally:
        env.close()
