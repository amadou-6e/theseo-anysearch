from __future__ import annotations

from pathlib import Path

import pytest

from theseo_anysearch.environments.action_spaces import ACTION_OFFSETS_26
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.experiments.native_extensions import compile_native_extension


def test_native_countdown_reward_resets_for_each_waypoint_segment() -> None:
    experiment = Path("usage", "experiments", "tune", "waypoint_route_segment_reward")
    manifest = compile_native_extension(experiment)
    env = VoxelEnv(
        {
            "grid_size": 8,
            "max_steps": 10,
            "trail_mode": False,
            "action_mode": "discrete_26",
            "obs_mode": "box",
            "box_radius": 1,
            "waypoint_route": {
                "start": (2, 2, 2),
                "waypoints": [(4, 2, 2), (5, 2, 2)],
            },
            "step_cost": 0.0,
            "collision_cost": 0.0,
            "goal_reward": 0.0,
            "distance_shaping": 0.0,
            "invalid_action_cost": 0.0,
            "construction_residual_weight": 0.0,
            "construction_overshoot_weight": 0.0,
            "custom_reward": "segment_countdown_goal",
            "custom_reward_parameters": {
                "additional_budget": 10.0,
                "minimum_reward": 1.0,
            },
            "native_extension_manifest": str(manifest),
            "task": {},
        }
    )
    positive_x = ACTION_OFFSETS_26.index((1, 0, 0))
    try:
        env.reset(seed=42)
        _, reward, terminated, truncated, info = env.step(positive_x)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert not info["waypoint_reached"]

        _, reward, terminated, truncated, info = env.step(positive_x)
        assert reward == pytest.approx(10.0)
        assert not terminated
        assert not truncated
        assert info["waypoint_reached"]
        assert not info["goal_reached"]
        assert info["reward_breakdown"] == {
            "segment_countdown_goal": pytest.approx(10.0)
        }

        _, reward, terminated, truncated, info = env.step(positive_x)
        assert reward == pytest.approx(10.0)
        assert terminated
        assert not truncated
        assert info["goal_reached"]
    finally:
        env.close()
