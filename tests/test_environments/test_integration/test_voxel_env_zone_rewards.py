"""Integration tests for distance-zone voxel rewards."""

from __future__ import annotations

import pytest

from ._voxel_validity_support import (
    ACTION_MINUS_Z,
    ACTION_PLUS_Y,
    ACTION_PLUS_Z,
    GOAL_REWARD,
    make_radial_test_env,
)


ZONE_REWARD_MIN = -1.0
ZONE_REWARD_MAX = -0.01


@pytest.mark.integration
class TestVoxelEnvZoneRewards:
    """Verify absolute-distance zone rewards stay dense and negative."""

    def test_zone_reward_gets_less_negative_when_closer_to_goal(self, tmp_path):
        toward_env = make_radial_test_env(
            tmp_path.joinpath("toward"),
            reward_overrides={
                "distance_reward_mode": "zone",
                "zone_reward_min": ZONE_REWARD_MIN,
                "zone_reward_max": ZONE_REWARD_MAX,
                "zone_reward_curve": "linear",
            },
        )
        away_env = make_radial_test_env(
            tmp_path.joinpath("away"),
            reward_overrides={
                "distance_reward_mode": "zone",
                "zone_reward_min": ZONE_REWARD_MIN,
                "zone_reward_max": ZONE_REWARD_MAX,
                "zone_reward_curve": "linear",
            },
        )
        neutral_env = make_radial_test_env(
            tmp_path.joinpath("neutral"),
            reward_overrides={
                "distance_reward_mode": "zone",
                "zone_reward_min": ZONE_REWARD_MIN,
                "zone_reward_max": ZONE_REWARD_MAX,
                "zone_reward_curve": "linear",
            },
        )

        toward_env.reset(seed=0)
        away_env.reset(seed=0)
        neutral_env.reset(seed=0)

        _, toward_reward, _, _, _ = toward_env.step(ACTION_PLUS_Z)
        _, away_reward, _, _, _ = away_env.step(ACTION_MINUS_Z)
        _, neutral_reward, _, _, _ = neutral_env.step(ACTION_PLUS_Y)

        assert ZONE_REWARD_MIN <= away_reward < neutral_reward < toward_reward < 0.0
        assert toward_reward <= ZONE_REWARD_MAX

    def test_goal_step_keeps_negative_zone_reward_plus_terminal_bonus(self, tmp_path):
        env = make_radial_test_env(
            tmp_path,
            reward_overrides={
                "distance_reward_mode": "zone",
                "zone_reward_min": ZONE_REWARD_MIN,
                "zone_reward_max": ZONE_REWARD_MAX,
                "zone_reward_curve": "linear",
            },
        )
        env.reset(seed=0)

        _, reward1, terminated1, _, _ = env.step(ACTION_PLUS_Z)
        _, reward2, terminated2, _, _ = env.step(ACTION_PLUS_Z)

        assert reward1 < 0.0
        assert reward2 == pytest.approx(GOAL_REWARD + ZONE_REWARD_MAX)
        assert terminated1 is False
        assert terminated2 is True

    def test_exponential_zone_curve_differs_from_linear(self, tmp_path):
        linear_env = make_radial_test_env(
            tmp_path.joinpath("linear"),
            reward_overrides={
                "distance_reward_mode": "zone",
                "zone_reward_min": ZONE_REWARD_MIN,
                "zone_reward_max": ZONE_REWARD_MAX,
                "zone_reward_curve": "linear",
            },
        )
        exp_env = make_radial_test_env(
            tmp_path.joinpath("exp"),
            reward_overrides={
                "distance_reward_mode": "zone",
                "zone_reward_min": ZONE_REWARD_MIN,
                "zone_reward_max": ZONE_REWARD_MAX,
                "zone_reward_curve": "exponential",
            },
        )
        linear_env.reset(seed=0)
        exp_env.reset(seed=0)

        _, linear_reward, _, _, _ = linear_env.step(ACTION_MINUS_Z)
        _, exp_reward, _, _, _ = exp_env.step(ACTION_MINUS_Z)

        assert exp_reward > linear_reward
        assert exp_reward < 0.0
