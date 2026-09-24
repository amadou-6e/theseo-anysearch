"""Scenario-provider reset lifecycle integration tests."""

from pathlib import Path

from theseo_anysearch.environments.action_spaces import ACTION_OFFSETS_26
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv


def test_evaluation_scenario_covers_all_adjacent_directions() -> None:
    source = Path(
        "usage",
        "experiments",
        "showcase",
        "scenario_extensions",
        "scenarios.py",
    ).resolve()
    env = VoxelEnv(
        {
            "grid_size": 32,
            "max_steps": 2,
            "agent_count": 1,
            "trail_mode": False,
            "action_mode": "discrete_26",
            "scenario_provider": "adjacent_goal_python",
            "scenario_parameters": {"seed_base": 142},
            "scenario_scope": "evaluation",
            "scenario_module_path": str(source),
            "geometry_boxes": [],
        }
    )
    assert "goal_direction" in env.observation_space.spaces

    observed = []
    for index in range(26):
        observation, info = env.reset(seed=142 + index)
        cursor = env._rust_env.cursor_pos()
        goal = env._rust_env.goal_pos()
        observed.append(tuple(goal[axis] - cursor[axis] for axis in range(3)))
        assert "goal_direction" in observation
        assert info["scenario"]["scenario_id"] == f"adjacent-{index:02d}"

    assert observed == list(ACTION_OFFSETS_26)
