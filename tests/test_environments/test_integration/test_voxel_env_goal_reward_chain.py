"""Integration tests for deterministic voxel goal-reward behavior."""

from __future__ import annotations

import pytest

from ._voxel_validity_support import (
    ACTION_MINUS_Z,
    ACTION_PLUS_X,
    ACTION_PLUS_Y,
    ACTION_PLUS_Z,
    COLLISION_COST,
    DISTANCE_SHAPING,
    GOAL_REWARD,
    STEP_COST,
    make_radial_test_env,
)


@pytest.mark.integration
class TestVoxelEnvGoalRewardChain:
    """Verify reward behavior for good, bad, neutral, and collision policies."""

    def test_goal_directed_policy_has_expected_reward_chain(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        env.reset(seed=0)

        _, reward1, terminated1, truncated1, _ = env.step(ACTION_PLUS_Z)
        _, reward2, terminated2, truncated2, _ = env.step(ACTION_PLUS_Z)

        assert reward1 == pytest.approx(STEP_COST + DISTANCE_SHAPING)
        assert reward2 == pytest.approx(STEP_COST + DISTANCE_SHAPING + GOAL_REWARD)
        assert terminated1 is False
        assert truncated1 is False
        assert terminated2 is True
        assert truncated2 is False
        assert reward1 + reward2 == pytest.approx(9.94)

    def test_goal_directed_step_outperforms_perpendicular_and_away(self, tmp_path):
        toward_env = make_radial_test_env(tmp_path.joinpath("toward"))
        away_env = make_radial_test_env(tmp_path.joinpath("away"))
        neutral_env = make_radial_test_env(tmp_path.joinpath("neutral"))

        toward_env.reset(seed=0)
        away_env.reset(seed=0)
        neutral_env.reset(seed=0)

        _, toward_reward, _, _, _ = toward_env.step(ACTION_PLUS_Z)
        _, away_reward, _, _, _ = away_env.step(ACTION_MINUS_Z)
        _, neutral_reward, _, _, _ = neutral_env.step(ACTION_PLUS_Y)

        assert toward_reward == pytest.approx(-0.03)
        assert away_reward == pytest.approx(-0.07)
        assert neutral_reward < toward_reward
        assert neutral_reward > away_reward

    def test_collision_step_applies_collision_penalty_without_goal_reward(self, tmp_path):
        geometry_boxes = [[5, 4, 4, 5, 4, 4]]
        env = make_radial_test_env(tmp_path, geometry_boxes=geometry_boxes)
        env.reset(seed=0)

        _, reward, terminated, truncated, _ = env.step(ACTION_PLUS_X)

        assert reward == pytest.approx(STEP_COST + COLLISION_COST)
        assert terminated is False
        assert truncated is False
        assert reward < STEP_COST

    def test_start_marker_is_filled_but_non_blocking(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        env.reset(seed=0)

        _, _, _, _, _ = env.step(ACTION_PLUS_Z)
        _, reward_back, terminated, truncated, _ = env.step(ACTION_MINUS_Z)

        assert reward_back == pytest.approx(STEP_COST - DISTANCE_SHAPING)
        assert terminated is False
        assert truncated is False
