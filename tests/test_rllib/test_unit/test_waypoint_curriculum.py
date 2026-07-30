from __future__ import annotations

from unittest.mock import patch

import pytest

from theseo_anysearch.models import EnvConfig, WaypointCurriculumConfig
from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
    WaypointCurriculum,
    broadcast_waypoints,
)


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


def test_enabled_curriculum_requires_initial_waypoints():
    with pytest.raises(ValueError, match="initial_start and initial_goal"):
        EnvConfig(waypoint_curriculum={"enabled": True})
