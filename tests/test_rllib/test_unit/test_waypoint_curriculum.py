from __future__ import annotations

from unittest.mock import patch

import pytest

from theseo_anysearch.models import EnvConfig, WaypointCurriculumConfig
from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
    WaypointCurriculum,
    broadcast_waypoints,
)
from theseo_anysearch.rllib.trainer.waypoint_routes import sample_route


def curriculum(**advance):
    return WaypointCurriculum(
        WaypointCurriculumConfig(
            enabled=True,
            initial_start=(9, 29, 1),
            initial_goal=(6, 7, 1),
            advance=advance,
        )
    )


def test_fixed_mode_never_advances():
    scheduler = curriculum(mode="fixed")
    assert scheduler.observe(10, 3) is False


def test_interval_mode_advances_without_success_requirement():
    scheduler = curriculum(
        mode="interval",
        interval_iterations=5,
        require_success=False,
    )
    assert scheduler.observe(4, 0) is False
    assert scheduler.observe(5, 0) is True


def test_interval_mode_waits_for_evaluation_success():
    scheduler = curriculum(
        mode="interval",
        interval_iterations=5,
        require_success=True,
        successes_required=1,
    )
    assert scheduler.observe(5, 0) is False
    assert scheduler.observe(9, 1) is False
    assert scheduler.observe(10, 0) is True


def test_success_mode_requires_configured_success_count():
    scheduler = curriculum(mode="success", successes_required=2)
    assert scheduler.observe(1, 1) is False
    assert scheduler.observe(2, 1) is True


def test_advance_resets_successes_and_records_transition():
    scheduler = curriculum(mode="success")
    assert scheduler.observe(1, 1) is True
    transition = scheduler.advance(1, (2, 3, 4), (7, 8, 9))
    assert scheduler.state.stage == 1
    assert scheduler.state.successes_in_stage == 0
    assert transition.from_stage == 0
    assert transition.to_stage == 1


def test_sampling_uses_reproducible_stage_seed():
    seeds: list[int] = []

    class FakeRustEnv:
        @staticmethod
        def cursor_pos():
            return (1, 2, 3)

        @staticmethod
        def goal_pos():
            return (4, 5, 6)

    class FakeEnv:
        def __init__(self, config):
            self.config = config
            self._rust_env = FakeRustEnv()

        def reset(self, *, seed=None):
            seeds.append(seed)

        def close(self):
            pass

    scheduler = WaypointCurriculum(
        WaypointCurriculumConfig(
            enabled=True,
            initial_start=(9, 29, 1),
            initial_goal=(6, 7, 1),
            seed=100,
        )
    )
    with patch(
        "theseo_anysearch.environments.gymnasium.voxel_env.VoxelEnv",
        FakeEnv,
    ):
        assert scheduler.sample({}) == ((1, 2, 3), (4, 5, 6))
        assert scheduler.sample({}) == ((1, 2, 3), (4, 5, 6))
    assert seeds == [101, 101]


def test_broadcast_queues_waypoints_on_every_environment():
    calls = []

    class FakeGroup:
        def foreach_env(self, function):
            class FakeEnv:
                def queue_waypoints(self, start, goal):
                    calls.append((start, goal))

            function(FakeEnv())
            function(FakeEnv())

    class FakeAlgo:
        env_runner_group = FakeGroup()

    broadcast_waypoints(FakeAlgo(), (1, 2, 3), (4, 5, 6))
    assert calls == [
        ((1, 2, 3), (4, 5, 6)),
        ((1, 2, 3), (4, 5, 6)),
    ]


def test_broadcast_queues_waypoints_through_modern_env_runners():
    calls = []

    class FakeVectorEnv:
        def call(self, method_name, *args):
            calls.append((method_name, args))

    class FakeRunner:
        env = FakeVectorEnv()

    class FakeGroup:
        def foreach_env_runner(self, function, *, local_env_runner):
            assert local_env_runner is False
            function(FakeRunner())

    class FakeAlgo:
        config = type(
            "Config",
            (),
            {"enable_env_runner_and_connector_v2": True},
        )()
        env_runner_group = FakeGroup()

    broadcast_waypoints(FakeAlgo(), (1, 2, 3), (4, 5, 6))

    assert calls == [
        ("queue_waypoints", ((1, 2, 3), (4, 5, 6))),
    ]

def test_enabled_curriculum_requires_initial_waypoints():
    with pytest.raises(ValueError, match="initial_start and initial_goal"):
        EnvConfig(waypoint_curriculum={"enabled": True})


def monotonic_curriculum(**difficulty):
    return WaypointCurriculum(
        WaypointCurriculumConfig(
            enabled=True,
            initial_start=(16, 16, 16),
            initial_goal=(18, 18, 18),
            seed=42,
            difficulty={
                "mode": "monotonic_distance",
                "distance_increment": 4.0,
                **difficulty,
            },
            advance={"mode": "success"},
        )
    )


def empty_grid(grid_size=32):
    return {
        "grid_size": grid_size,
        "geometry_boxes": [],
        "stl_path": None,
        "stl_paths": None,
        "geometry_pool": None,
    }


def test_configured_route_stages_cover_exact_distance_schedule():
    scheduler = WaypointCurriculum(
        WaypointCurriculumConfig(
            enabled=True,
            completion_mode="continue_route",
            initial_start=(16, 16, 16),
            route_length={"mode": "fixed", "distance": 20},
            difficulty={
                "mode": "segment_distance",
                "initial_distance": 1,
                "distance_increment": 2,
                "maximum_distance": 6,
            },
        ),
        empty_grid(),
    )

    with patch(
        "theseo_anysearch.rllib.trainer.waypoint_curriculum.sample_route",
        wraps=sample_route,
    ) as route_sampler:
        routes = scheduler.configured_route_stages(empty_grid())

    assert [call.kwargs["segment_distance"] for call in route_sampler.call_args_list] == [
        1,
        3,
        5,
        6,
    ]
    assert [len(route.waypoints) for route in routes] == [20, 7, 4, 4]


def test_route_for_stage_uses_collection_seed_for_new_routes():
    scheduler = WaypointCurriculum(
        WaypointCurriculumConfig(
            enabled=True,
            completion_mode="continue_route",
            initial_start=(16, 16, 16),
            route_length={"mode": "fixed", "distance": 20},
            difficulty={
                "mode": "segment_distance",
                "initial_distance": 3,
                "maximum_distance": 3,
            },
        ),
        empty_grid(),
    )

    first = scheduler.route_for_stage(empty_grid(), 0, seed=1000)
    second = scheduler.route_for_stage(empty_grid(), 0, seed=1001)

    assert first != second
    assert len(first.waypoints) == len(second.waypoints) == 7


def distance(pair):
    import math

    return math.dist(*pair)


def test_monotonic_distance_keeps_start_centered_within_grid_radius():
    scheduler = monotonic_curriculum()

    first = scheduler.sample(empty_grid())
    scheduler.advance(1, *first)
    second = scheduler.sample(empty_grid())

    assert first[0] == (16, 16, 16)
    assert second[0] == (16, 16, 16)
    assert distance(second) >= distance(first)


def test_monotonic_distance_randomizes_both_waypoints_beyond_grid_radius():
    scheduler = monotonic_curriculum(distance_increment=20.0)

    pair = scheduler.sample(empty_grid())

    assert pair[0] != (16, 16, 16)
    assert all(1 <= coordinate <= 32 for waypoint in pair for coordinate in waypoint)
    assert distance(pair) > 15.0


def test_monotonic_distance_never_decreases_and_caps_at_grid_diagonal():
    import math

    scheduler = monotonic_curriculum(distance_increment=8.0)
    distances = [distance((scheduler.state.start, scheduler.state.goal))]
    for iteration in range(1, 8):
        pair = scheduler.sample(empty_grid())
        distances.append(distance(pair))
        scheduler.advance(iteration, *pair)

    assert distances == sorted(distances)
    assert distances[-1] <= math.sqrt(3.0) * 31


def test_monotonic_distance_sampling_is_reproducible():
    first = monotonic_curriculum(distance_increment=20.0)
    second = monotonic_curriculum(distance_increment=20.0)

    assert first.sample(empty_grid()) == second.sample(empty_grid())


def test_monotonic_distance_honors_configured_maximum():
    scheduler = monotonic_curriculum(
        distance_increment=20.0,
        maximum_distance=12.0,
    )

    pair = scheduler.sample(empty_grid())

    assert distance(pair) <= 12.5


def test_monotonic_distance_rejects_static_geometry():
    scheduler = monotonic_curriculum()
    config = empty_grid()
    config["geometry_boxes"] = [[1, 1, 1, 2, 2, 2]]

    with pytest.raises(ValueError, match="requires empty geometry"):
        scheduler.sample(config)

def test_monotonic_distance_rejects_maximum_below_initial_distance():
    with pytest.raises(ValueError, match="below the initial waypoint distance"):
        WaypointCurriculumConfig(
            enabled=True,
            initial_start=(1, 1, 1),
            initial_goal=(5, 5, 5),
            difficulty={
                "mode": "monotonic_distance",
                "maximum_distance": 2.0,
            },
        )


def test_monotonic_distance_resamples_directions_at_distance_cap():
    scheduler = monotonic_curriculum(
        distance_increment=20.0,
        maximum_distance=12.0,
    )

    first = scheduler.sample(empty_grid())
    scheduler.advance(1, *first)
    second = scheduler.sample(empty_grid())

    assert distance(second) >= distance(first)
    assert second != first
