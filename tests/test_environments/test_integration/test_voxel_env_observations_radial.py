"""Integration tests for deterministic radial voxel observations."""

from __future__ import annotations

import numpy as np
import pytest

from ._voxel_validity_support import (
    ACTION_MINUS_Z,
    BLOCK_KIND_FILLED,
    BLOCK_KIND_GOAL,
    ACTION_PLUS_Y,
    ACTION_PLUS_Z,
    GOAL,
    MAX_STEPS,
    RAY_INDEX_PLUS_X,
    RAY_TYPE_INDEX_PLUS_Z,
    START,
    make_radial_test_env,
    normalized_cursor,
    normalized_goal_distance,
    normalized_goal_direction,
)


@pytest.mark.integration
class TestVoxelEnvObservationsRadial:
    """Verify each observation field responds correctly in deterministic cases."""

    def test_radial_observation_keys_and_shapes_match_current_mode(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        obs, _ = env.reset(seed=0)

        assert set(obs) == {
            "steps_remaining",
            "voxel_count",
            "goal_distance",
            "goal_direction",
            "cursor_pos",
            "ray_hits",
            "ray_hit_types",
        }
        assert obs["steps_remaining"].shape == (1,)
        assert obs["voxel_count"].shape == (1,)
        assert obs["goal_distance"].shape == (1,)
        assert obs["goal_direction"].shape == (3,)
        assert obs["cursor_pos"].shape == (3,)
        assert obs["ray_hits"].shape == (26,)
        assert obs["ray_hit_types"].shape == (26,)

    def test_steps_remaining_decreases_after_each_step(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(ACTION_PLUS_Z)
        obs2, *_ = env.step(ACTION_PLUS_Z)

        assert obs0["steps_remaining"][0] == pytest.approx(1.0)
        assert obs1["steps_remaining"][0] == pytest.approx((MAX_STEPS - 1) / MAX_STEPS)
        assert obs2["steps_remaining"][0] == pytest.approx((MAX_STEPS - 2) / MAX_STEPS)

    def test_goal_distance_decreases_toward_goal_and_increases_away(self, tmp_path):
        toward_env = make_radial_test_env(tmp_path.joinpath("toward"))
        away_env = make_radial_test_env(tmp_path.joinpath("away"))
        neutral_env = make_radial_test_env(tmp_path.joinpath("neutral"))

        obs0, _ = toward_env.reset(seed=0)
        obs_toward, *_ = toward_env.step(ACTION_PLUS_Z)

        away_env.reset(seed=0)
        obs_away, *_ = away_env.step(ACTION_MINUS_Z)

        neutral_env.reset(seed=0)
        obs_neutral, *_ = neutral_env.step(ACTION_PLUS_Y)

        assert obs0["goal_distance"][0] == pytest.approx(normalized_goal_distance(2))
        assert obs_toward["goal_distance"][0] == pytest.approx(normalized_goal_distance(1))
        assert obs_away["goal_distance"][0] == pytest.approx(normalized_goal_distance(3))
        assert obs_neutral["goal_distance"][0] == pytest.approx(normalized_goal_distance(3))

    def test_cursor_pos_tracks_deterministic_moves(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(ACTION_PLUS_Z)
        obs2, *_ = env.step(ACTION_PLUS_Y)

        assert tuple(obs0["cursor_pos"]) == pytest.approx(normalized_cursor(START))
        assert tuple(obs1["cursor_pos"]) == pytest.approx(normalized_cursor((4, 4, 5)))
        assert tuple(obs2["cursor_pos"]) == pytest.approx(normalized_cursor((4, 5, 5)))

    def test_goal_direction_tracks_deterministic_moves(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(ACTION_PLUS_Z)
        obs2, *_ = env.step(ACTION_PLUS_Y)

        assert tuple(obs0["goal_direction"]) == pytest.approx(normalized_goal_direction(START, GOAL))
        assert tuple(obs1["goal_direction"]) == pytest.approx(
            normalized_goal_direction((4, 4, 5), GOAL)
        )
        assert tuple(obs2["goal_direction"]) == pytest.approx(
            normalized_goal_direction((4, 5, 5), GOAL)
        )

    def test_voxel_count_increases_on_successful_moves(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(ACTION_PLUS_Z)
        obs2, *_ = env.step(ACTION_PLUS_Y)

        assert obs0["voxel_count"][0] == pytest.approx(0.0)
        assert obs1["voxel_count"][0] == pytest.approx(1.0)
        assert obs2["voxel_count"][0] == pytest.approx(2.0)

    def test_ray_hits_show_adjacent_geometry(self, tmp_path):
        geometry_boxes = [[5, 4, 4, 5, 4, 4]]
        env = make_radial_test_env(tmp_path, geometry_boxes=geometry_boxes)
        obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(ACTION_PLUS_Z)

        assert obs0["ray_hits"][RAY_INDEX_PLUS_X] == pytest.approx(1.0)
        assert np.all((obs1["ray_hits"] >= 0.0) & (obs1["ray_hits"] <= 1.0))

    def test_ray_hit_types_encode_goal_and_generic_filled_voxels(self, tmp_path):
        env = make_radial_test_env(tmp_path)
        obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(ACTION_PLUS_Z)

        assert obs0["ray_hit_types"][RAY_TYPE_INDEX_PLUS_Z] == pytest.approx(BLOCK_KIND_GOAL)
        assert obs1["ray_hit_types"][RAY_TYPE_INDEX_PLUS_Z] == pytest.approx(BLOCK_KIND_GOAL)
        assert np.all(obs0["ray_hit_types"] >= 0.0)

        geometry_env = make_radial_test_env(
            tmp_path.joinpath("geometry"),
            geometry_boxes=[[5, 4, 4, 5, 4, 4]],
        )
        geometry_obs, _ = geometry_env.reset(seed=0)
        assert geometry_obs["ray_hit_types"][RAY_INDEX_PLUS_X] == pytest.approx(BLOCK_KIND_FILLED)
