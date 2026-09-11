from __future__ import annotations

import pytest

from theseo_anysearch.models import WaypointCurriculumConfig, WaypointRouteLengthConfig
from theseo_anysearch.rllib.trainer.curriculum.waypoint import CurriculumController
from theseo_anysearch.rllib.trainer.waypoint_curriculum import WaypointCurriculum
from theseo_anysearch.rllib.trainer.waypoint_routes import (
    action_step_distance,
    route_distance,
    sample_route,
    segment_lengths,
)


def route_curriculum_config(
    *,
    initial_distance: int = 1,
    distance_increment: float = 2.0,
    maximum_distance: float = 20.0,
) -> WaypointCurriculumConfig:
    return WaypointCurriculumConfig.model_validate({
        "enabled": True,
        "completion_mode": "continue_route",
        "initial_start": [16, 16, 16],
        "route_length": {"mode": "fixed", "distance": 24},
        "difficulty": {
            "mode": "segment_distance",
            "initial_distance": initial_distance,
            "distance_increment": distance_increment,
            "maximum_distance": maximum_distance,
        },
        "advance": {"mode": "success"},
    })


def route_environment() -> dict[str, object]:
    return {
        "grid_size": 32,
        "max_steps": 32,
        "action_mode": "discrete_18",
    }


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


@pytest.mark.parametrize(
    ("initial_distance", "increment", "maximum_distance", "maximum_stage"),
    [
        (2, 2.0, 20.0, 9),
        (1, 2.0, 20.0, 10),
        (20, 2.0, 20.0, 0),
        (1, 0.1, 1.3, 3),
    ],
)
def test_continue_route_curriculum_computes_final_distinct_stage(
    initial_distance,
    increment,
    maximum_distance,
    maximum_stage,
):
    curriculum = WaypointCurriculum(
        route_curriculum_config(
            initial_distance=initial_distance,
            distance_increment=increment,
            maximum_distance=maximum_distance,
        ),
        route_environment(),
    )

    assert curriculum.maximum_stage == maximum_stage


def test_terminal_route_curriculum_does_not_add_duplicate_capped_stage():
    curriculum = WaypointCurriculum(route_curriculum_config(), route_environment())
    for iteration in range(1, 11):
        assert curriculum.observe(iteration, 1) is True
        curriculum.advance_stage(iteration, curriculum.sample_stage(route_environment()))

    assert curriculum.state.stage == 10
    assert len(curriculum.stages()) == 11
    assert curriculum.terminal is True
    assert curriculum.observe(11, 1) is False
    assert curriculum.state.successes_in_stage == 0
    with pytest.raises(RuntimeError, match="terminal"):
        curriculum.sample_stage(route_environment())
    with pytest.raises(RuntimeError, match="terminal"):
        curriculum.advance_stage(11, curriculum.stages()[-1])


def test_restored_route_curriculum_is_clamped_to_final_distinct_stage():
    curriculum = WaypointCurriculum(
        route_curriculum_config(maximum_distance=24),
        route_environment(),
    )
    for iteration in range(1, 13):
        stage = curriculum._sample_route(route_environment(), iteration)
        curriculum.state.stage = iteration - 1
        curriculum.state.successes_in_stage = 4
        curriculum.advance_stage(iteration, stage)
    curriculum.config = route_curriculum_config()
    curriculum.state.stage_evaluations = {
        stage: {"attempts": 3, "successes": 3}
        for stage in range(13)
    }

    assert curriculum.clamp_restored_state() is True
    assert curriculum.state.stage == 10
    assert len(curriculum.state.transitions) == 10
    assert len(curriculum.stages()) == 11
    assert set(curriculum.state.stage_evaluations) == set(range(11))
    assert curriculum.state.successes_in_stage == 0
    assert curriculum.state.goal == curriculum.state.transitions[-1].goal
    assert curriculum.clamp_restored_state() is False


def test_curriculum_stage_metrics_expose_terminal_and_maximum_stage():
    curriculum = WaypointCurriculum(route_curriculum_config(), route_environment())
    controller = CurriculumController.__new__(CurriculumController)
    controller.curriculum = curriculum

    assert controller.stage_metric() == {
        "curriculum/stage": 0.0,
        "curriculum/terminal": 0.0,
        "curriculum/max_stage": 10.0,
    }

    curriculum.state.stage = 10
    assert controller.stage_metric()["curriculum/terminal"] == 1.0


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
        assert info["route_waypoints_reached"] == 1
        assert info["route_waypoint_completion_fraction"] == 0.5
        assert info["route_waypoints_remaining"] == 0
        assert observation["goal_direction"][0] > 0

        _, _, terminated, truncated, info = env.step(positive_x)
        assert terminated
        assert not truncated
        assert info["goal_reached"] is True
        assert info["route_waypoints_reached"] == 2
        assert info["route_waypoint_completion_fraction"] == 1.0
    finally:
        env.close()


def test_sample_route_honors_non_cubic_extent_deterministically():
    from theseo_anysearch.rllib.trainer.waypoint_routes import sample_route

    kwargs = {
        "start": (3, 2, 2),
        "total_distance": 8,
        "segment_distance": 2,
        "extent": (12, 4, 3),
        "action_mode": "discrete_26",
        "seed": 91,
    }
    first = sample_route(**kwargs)
    second = sample_route(**kwargs)
    assert first == second
    assert all(
        1 <= point[0] <= 12 and 1 <= point[1] <= 4 and 1 <= point[2] <= 3
        for point in first.waypoints
    )
