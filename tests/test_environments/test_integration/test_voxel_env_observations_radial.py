"""Integration tests for deterministic radial voxel observations."""

from __future__ import annotations

import numpy as np
import pytest

from ._voxel_validity_support import (
    ACTION_MINUS_X,
    ACTION_MINUS_Y,
    ACTION_MINUS_Z,
    BLOCK_KIND_BOUNDARY,
    BLOCK_KIND_FILLED,
    BLOCK_KIND_GOAL,
    BLOCK_KIND_OCCUPIED,
    COLLISION_COST,
    ACTION_PLUS_Y,
    ACTION_PLUS_Z,
    GOAL,
    MAX_STEPS,
    RAY_INDEX_MINUS_X,
    RAY_INDEX_PLUS_X,
    RAY_TYPE_INDEX_PLUS_Z,
    START,
    ACTION_PLUS_X,
    GRID_SIZE,
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
        assert np.linalg.norm(obs0["goal_direction"]) == pytest.approx(1.0)
        assert np.linalg.norm(obs1["goal_direction"]) == pytest.approx(1.0)
        assert np.linalg.norm(obs2["goal_direction"]) == pytest.approx(1.0)

    def test_goal_direction_does_not_encode_goal_distance(self, tmp_path):
        near = make_radial_test_env(
            tmp_path.joinpath("near"), start=(4, 4, 4), goal=(5, 4, 4)
        )
        far = make_radial_test_env(
            tmp_path.joinpath("far"), start=(4, 4, 4), goal=(20, 4, 4)
        )

        near_obs, _ = near.reset(seed=0)
        far_obs, _ = far.reset(seed=0)

        assert tuple(near_obs["goal_direction"]) == pytest.approx((1.0, 0.0, 0.0))
        assert tuple(far_obs["goal_direction"]) == pytest.approx((1.0, 0.0, 0.0))

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

    def test_ray_hit_types_encode_goal_and_static_occupied_voxels(self, tmp_path):
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
        assert geometry_obs["ray_hit_types"][RAY_INDEX_PLUS_X] == pytest.approx(BLOCK_KIND_OCCUPIED)

    def test_ray_hit_types_encode_agent_filled_trail_voxels(self, tmp_path):
        env = make_radial_test_env(tmp_path, start=(4, 4, 4), goal=(4, 4, 7))
        env.reset(seed=0)
        env.step(ACTION_PLUS_Z)
        obs, *_ = env.step(ACTION_PLUS_Z)

        assert obs["ray_hit_types"][ACTION_MINUS_Z] == pytest.approx(BLOCK_KIND_FILLED)

    def test_ray_hits_show_grid_boundary_as_blocked_space(self, tmp_path):
        env = make_radial_test_env(tmp_path, start=(1, 4, 4), goal=GOAL)
        obs, _ = env.reset(seed=0)

        assert obs["ray_hits"][RAY_INDEX_MINUS_X] == pytest.approx(1.0)
        assert obs["ray_hit_types"][RAY_INDEX_MINUS_X] == pytest.approx(BLOCK_KIND_BOUNDARY)

    def test_collision_actions_are_visible_in_previous_observation(self, tmp_path):
        env = make_radial_test_env(
            tmp_path,
            start=(2, 4, 4),
            goal=(2, 4, 8),
            geometry_boxes=[[3, 4, 4, 3, 4, 4]],
        )
        obs, _ = env.reset(seed=0)

        collision_checks = [
            (ACTION_PLUS_X, BLOCK_KIND_OCCUPIED, "static geometry"),
            (ACTION_MINUS_X, BLOCK_KIND_BOUNDARY, "grid boundary"),
            (ACTION_MINUS_Y, BLOCK_KIND_FILLED, "agent-filled trail"),
        ]

        observed_collisions = []
        steps_taken = 0

        for action, expected_type, label in collision_checks:
            if label == "grid boundary":
                obs, *_ = env.step(ACTION_MINUS_X)
                steps_taken += 1
            elif label == "agent-filled trail":
                obs, *_ = env.step(ACTION_PLUS_Y)
                steps_taken += 1

            cursor_before = obs["cursor_pos"].copy()
            ray_hit_before = obs["ray_hits"][action]
            ray_type_before = obs["ray_hit_types"][action]

            obs, *_ = env.step(action)
            steps_taken += 1
            cursor_after = obs["cursor_pos"].copy()

            if np.array_equal(cursor_before, cursor_after):
                observed_collisions.append(label)
                assert ray_hit_before == pytest.approx(1.0), label
                assert ray_type_before == pytest.approx(expected_type), label
                assert 0.0 <= ray_type_before <= 1.0, label

        assert observed_collisions == [
            "static geometry",
            "grid boundary",
            "agent-filled trail",
        ]
        assert steps_taken == 5
        assert obs["steps_remaining"][0] == pytest.approx((MAX_STEPS - steps_taken) / MAX_STEPS)

    def test_random_collision_actions_are_visible_for_100_steps(self, tmp_path):
        env = make_radial_test_env(
            tmp_path,
            start=(2, 2, 2),
            goal=(GRID_SIZE, GRID_SIZE, GRID_SIZE),
            geometry_boxes=[
                [3, 2, 2, 3, 2, 2],
                [2, 4, 2, 2, 4, 2],
                [2, 2, 5, 2, 2, 5],
                [5, 5, 5, 7, 5, 5],
            ],
        )
        rng = np.random.default_rng(20260413)
        obs, _ = env.reset(seed=0)

        collisions = 0

        for step_index in range(100):
            action = int(rng.integers(0, 26))
            cursor_before = obs["cursor_pos"].copy()
            ray_hit_before = float(obs["ray_hits"][action])
            ray_type_before = float(obs["ray_hit_types"][action])
            obs, reward, terminated, truncated, _ = env.step(action)
            cursor_after = obs["cursor_pos"].copy()

            if np.array_equal(cursor_before, cursor_after):
                collisions += 1
                assert ray_hit_before > 0.0, step_index
                assert ray_type_before == pytest.approx(
                    BLOCK_KIND_OCCUPIED
                ) or ray_type_before == pytest.approx(
                    BLOCK_KIND_BOUNDARY
                ) or ray_type_before == pytest.approx(
                    BLOCK_KIND_FILLED
                ), step_index
                assert reward <= COLLISION_COST, step_index

            if step_index < 99:
                assert not terminated
                assert not truncated

        assert collisions > 0
        assert obs["steps_remaining"][0] == pytest.approx(0.0)
