"""Obstacle-world waypoint preflight exercises the compiled regional backend."""

from usage.experiments.train.large_world_obstacle_pilot.preflight import (
    EXTENT,
    SOURCES,
    direct_path_is_free,
    preflight,
)


def test_direct_waypoint_actions_detect_an_obstacle() -> None:
    class World:
        @staticmethod
        def world_occupied(position):
            return position == (2, 1, 1)

    assert not direct_path_is_free(World(), (1, 1, 1), (3, 1, 1))
    assert direct_path_is_free(World(), (1, 2, 1), (3, 2, 1))


def test_compiled_obstacle_routes_require_planning(tmp_path) -> None:
    report = preflight(tmp_path / "worlds", samples_per_stage=1)

    assert report["extent"] == EXTENT
    assert len(report["sources"]) == len(SOURCES)
    assert len(report["routes"]) == 11
    assert any(route["astar_feasible"] for route in report["routes"])
    assert any(
        route["astar_feasible"] and not route["direct_path_free"]
        for route in report["routes"]
    )
    assert report["astar_detour_replay"]["success"]
