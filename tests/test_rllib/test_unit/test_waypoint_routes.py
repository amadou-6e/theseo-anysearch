from __future__ import annotations

import pytest

from theseo_anysearch.models import WaypointCurriculumConfig, WaypointRouteLengthConfig
from theseo_anysearch.rllib.trainer.waypoint_curriculum import WaypointCurriculum
from theseo_anysearch.rllib.trainer.waypoint_routes import (
    action_step_distance,
    route_distance,
    sample_route,
    segment_lengths,
)


def test_route_length_modes_resolve_exactly():
    assert WaypointRouteLengthConfig(mode="fixed", distance=150).resolve(200) == 150
    assert WaypointRouteLengthConfig(mode="fraction", fraction=0.75).resolve(200) == 150


@pytest.mark.parametrize(
    ("action_mode", "expected"),
    [("discrete_6", 6), ("discrete_18", 3), ("discrete_26", 3), ("vector_3", 3)],
)
def test_action_step_distance_matches_action_graph(action_mode, expected):
    assert action_step_distance((1, 1, 1), (4, 3, 2), action_mode) == expected


def test_last_segment_is_exact_residual():
    assert segment_lengths(17, 5) == (5, 5, 5, 2)


def test_sampled_route_has_exact_total_and_segment_lengths():
    route = sample_route(
        start=(16, 16, 16),
        total_distance=17,
        segment_distance=5,
        grid_size=32,
        action_mode="discrete_18",
        seed=42,
    )
    points = (route.start, *route.waypoints)
    distances = tuple(
        action_step_distance(points[index], points[index + 1], "discrete_18")
        for index in range(len(points) - 1)
    )
    assert distances == (5, 5, 5, 2)
    assert route_distance(route, "discrete_18") == 17


def test_continue_route_curriculum_generates_initial_stage():
    config = WaypointCurriculumConfig.model_validate({
        "enabled": True,
        "completion_mode": "continue_route",
        "initial_start": [16, 16, 16],
        "route_length": {"mode": "fraction", "fraction": 0.75},
        "difficulty": {
            "mode": "segment_distance",
            "initial_distance": 2,
            "distance_increment": 2,
            "maximum_distance": 20,
        },
    })
    curriculum = WaypointCurriculum(config, {
        "grid_size": 32,
        "max_steps": 200,
        "action_mode": "discrete_18",
    })
    assert route_distance(curriculum._initial_route, "discrete_18") == 150
    assert all(
        action_step_distance(a, b, "discrete_18") == 2
        for a, b in zip(
            (curriculum._initial_route.start, *curriculum._initial_route.waypoints[:-1]),
            curriculum._initial_route.waypoints,
        )
    )


def test_environment_continues_at_intermediate_waypoint():
    from theseo_anysearch.environments.action_spaces import ACTION_OFFSETS_26
    from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

    env = VoxelEnv({
        "grid_size": 8,
        "max_steps": 10,
        "trail_mode": False,
        "action_mode": "discrete_26",
        "obs_mode": "box",
        "box_radius": 1,
        "waypoint_route": {
            "start": (2, 2, 2),
            "waypoints": [(3, 2, 2), (4, 2, 2)],
        },
        "task": {},
    })
    try:
        env.reset(seed=42)
        positive_x = ACTION_OFFSETS_26.index((1, 0, 0))
        observation, _, terminated, truncated, info = env.step(positive_x)
        assert not terminated
        assert not truncated
        assert info["waypoint_reached"] is True
        assert info["goal_reached"] is False
        assert info["route_waypoints_remaining"] == 0
        assert observation["goal_direction"][0] > 0

        _, _, terminated, truncated, info = env.step(positive_x)
        assert terminated
        assert not truncated
        assert info["goal_reached"] is True
    finally:
        env.close()
