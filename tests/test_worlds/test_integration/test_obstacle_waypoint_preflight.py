"""Obstacle-world waypoint preflight exercises the compiled regional backend."""

import json

from usage.experiments.train.large_world_obstacle_pilot.preflight import (
    EXTENT,
    SOURCES,
    direct_path_is_free,
    preflight,
)
from usage.experiments.train.large_world_obstacle_pilot.preview import write_preview_files


def test_direct_waypoint_actions_detect_an_obstacle() -> None:
    class World:
        @staticmethod
        def world_occupied(position):
            return position == (2, 1, 1)

    assert not direct_path_is_free(World(), (1, 1, 1), (3, 1, 1))
    assert direct_path_is_free(World(), (1, 2, 1), (3, 2, 1))


def test_compiled_obstacle_routes_require_planning(tmp_path) -> None:
    report = preflight(tmp_path / "worlds", samples_per_stage=1, region_indices=(0,))

    assert report["extent"] == EXTENT
    assert len(report["sources"]) == len(SOURCES)
    assert report["logical_cells"] == 4_294_967_296
    assert max(source.maximum_inclusive[0] for source in SOURCES) > 3900
    assert max(source.maximum_inclusive[1] for source in SOURCES) > 1900
    assert max(source.maximum_inclusive[2] for source in SOURCES) > 400
    assert len(report["routes"]) == 11
    assert any(route["astar_feasible"] for route in report["routes"])
    assert any(
        route["astar_feasible"] and not route["direct_path_free"]
        for route in report["routes"]
    )
    assert report["astar_detour_replay"]["success"]
    previews = write_preview_files(report, tmp_path / "previews")
    assert len(previews) == 1
    preview = json.loads(previews[0].read_text(encoding="utf-8"))
    assert preview["world"]["identity_sha256"] == report["world_identity"]
    assert preview["episode"]["start_pos"] == list(report["route_regions"][0]["start"])
    assert (previews[0].parent / preview["world"]["manifest_path"]).resolve().is_file()
