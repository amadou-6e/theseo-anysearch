"""Deterministic tests for the versioned voxel task contract."""

from __future__ import annotations

import pytest

from tests.test_environments.test_integration._voxel_validity_support import (
    ACTION_PLUS_X,
    ACTION_PLUS_Y,
    ACTION_PLUS_Z,
    GOAL_REWARD,
    START,
    make_radial_test_env,
)


@pytest.mark.integration
class TestTaskContract:
    """Verify point/set success, reward attribution, and termination semantics."""

    def test_point_goal_reports_components_and_success_reason(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        _, reset_info = env.reset(seed=0)

        _, reward, terminated, truncated, info = env.step(ACTION_PLUS_Z)
        assert reset_info["initial_goal_distance"] == pytest.approx(2.0)
        assert terminated is False
        assert truncated is False
        assert info["termination_reason"] == "in_progress"
        assert reward == pytest.approx(sum(info["reward_breakdown"].values()))

        _, _, terminated, truncated, info = env.step(ACTION_PLUS_Z)
        assert terminated is True
        assert truncated is False
        assert info["goal_reached"] is True
        assert info["termination_reason"] == "success"
        assert info["reward_breakdown"]["success"] == pytest.approx(GOAL_REWARD)
        assert info["minimum_goal_distance"] == pytest.approx(0.0)

    def test_target_voxel_set_accepts_any_member(self, tmp_path):
        target = (START[0], START[1] + 1, START[2])
        env = make_radial_test_env(
            tmp_path,
            reward_overrides={
                "task": {
                    "version": 1,
                    "goal": {
                        "type": "target_voxel_set",
                        "voxels": [target, (target[0], target[1] + 1, target[2])],
                    },
                }
            },
        )
        env.reset(seed=0)

        _, _, terminated, truncated, info = env.step(ACTION_PLUS_Y)

        assert terminated is True
        assert truncated is False
        assert info["goal_reached"] is True

    def test_step_limit_is_truncation_with_explicit_reason(self, tmp_path):
        env = make_radial_test_env(
            tmp_path,
            reward_overrides={"max_steps": 1, "distance_shaping": 0.0},
        )
        env.reset(seed=0)

        _, reward, terminated, truncated, info = env.step(ACTION_PLUS_Y)

        assert terminated is False
        assert truncated is True
        assert info["termination_reason"] == "step_limit"
        assert info["reward_breakdown"]["distance_progress"] == 0.0
        assert info["unshaped_reward"] == pytest.approx(reward)

    def test_consecutive_collision_limit_terminates_and_resets_after_movement(self, tmp_path):
        env = make_radial_test_env(
            tmp_path,
            geometry_boxes=[[5, 4, 4, 5, 4, 5]],
            reward_overrides={
                "task": {"max_consecutive_collisions": 3},
            },
        )
        env.reset(seed=0)

        _, _, terminated, truncated, info = env.step(ACTION_PLUS_X)
        assert terminated is False
        assert truncated is False
        assert info["consecutive_collisions"] == 1

        _, _, terminated, _, info = env.step(ACTION_PLUS_Z)
        assert terminated is False
        assert info["consecutive_collisions"] == 0

        for expected_count in (1, 2):
            _, _, terminated, truncated, info = env.step(ACTION_PLUS_X)
            assert terminated is False
            assert truncated is False
            assert info["consecutive_collisions"] == expected_count

        _, _, terminated, truncated, info = env.step(ACTION_PLUS_X)
        assert terminated is False
        assert truncated is True
        assert info["goal_reached"] is False
        assert info["termination_reason"] == "collision_limit"
        assert info["consecutive_collisions"] == 3

    def test_invalid_action_is_separate_from_collision(self, tmp_path):
        env = make_radial_test_env(
            tmp_path,
            reward_overrides={"invalid_action_cost": -0.7, "collision_cost": -0.5},
        )
        env.reset(seed=0)

        _, _, _, _, info = env.step(99)

        assert info["invalid_action"] is True
        assert info["collision"] is False
        assert info["reward_breakdown"]["invalid_action"] == pytest.approx(-0.7)
        assert info["reward_breakdown"]["collision"] == 0.0

    def test_construction_terms_report_residual_and_overshoot(self, tmp_path):
        target = (START[0], START[1], START[2] + 1)
        env = make_radial_test_env(
            tmp_path,
            reward_overrides={
                "construction_residual_weight": 0.2,
                "construction_overshoot_weight": 0.3,
                "task": {"construction_target_voxels": [target]},
            },
        )
        env.reset(seed=0)

        _, _, _, _, info = env.step(ACTION_PLUS_Z)

        assert info["construction_residual"] == 0
        assert info["construction_overshoot"] == 0
        assert info["reward_breakdown"]["construction_residual"] == 0.0
        assert info["reward_breakdown"]["construction_overshoot"] == 0.0
