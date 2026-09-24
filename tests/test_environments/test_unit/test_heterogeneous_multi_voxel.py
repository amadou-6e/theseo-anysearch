"""Tests for heterogeneous PettingZoo voxel agents."""

import numpy as np

from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv


def config() -> dict:
    """Return a dependency-free two-agent capture environment."""
    pipeline = {
        "action_predicates": [
            {"name": "valid_action"},
            {"name": "bounds"},
            {"name": "unoccupied"},
        ],
        "action_outcomes": [{"name": "cursor_movement"}],
        "action_history_length": 8,
    }
    return {
        "agent_count": 2,
        "max_steps": 1,
        "grid_size": 16,
        "box_radius": 2,
        "agents": [
            {"id": "hunted", "policy": "hunted", "action_mode": "discrete_6", "start": [10, 1, 1], **pipeline},
            {"id": "hunter", "policy": "hunter", "action_mode": "discrete_18", "start": [1, 1, 1], **pipeline},
        ],
        "hunter_and_hunted": {
            "hunter": "hunter",
            "hunted": "hunted",
            "capture_distance": 1,
            "hunter_capture_reward": 1.0,
            "hunted_escape_reward": 2.0,
        },
        "step_cost": 0.0,
        "distance_shaping": 0.0,
    }


def test_named_agents_have_independent_action_spaces_and_relative_observation() -> None:
    env = MultiVoxelEnv(config())
    observations, _ = env.reset(seed=42)
    assert env.possible_agents == ["hunted", "hunter"]
    assert env.action_space("hunted").n == 6
    assert env.action_space("hunter").n == 18
    assert observations["hunted"]["other_agent_vectors"].shape == (3,)
    np.testing.assert_allclose(
        observations["hunted"]["other_agent_vectors"],
        np.array([-0.6, 0.0, 0.0], dtype=np.float32),
    )


def test_timeout_reward_is_fanned_out_to_hunted_policy() -> None:
    env = MultiVoxelEnv(config())
    env.reset(seed=42)
    _, rewards, terminations, _, _ = env.step({"hunted": 0, "hunter": 0})
    assert all(terminations.values())
    assert rewards == {"hunted": 2.0, "hunter": 0.0}
