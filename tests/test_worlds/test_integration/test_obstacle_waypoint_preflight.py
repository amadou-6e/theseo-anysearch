"""Obstacle-world waypoint preflight exercises the compiled regional backend."""

import json

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic.voxel.astar.standard import VoxelAStarOracle
from usage.experiments.train.large_world_obstacle_pilot.preflight import (
    EXTENT,
    PORTAL_CENTER,
    PORTAL_WALLS,
    SOURCES,
    direct_path_is_free,
    preflight,
    wall_sources,
)
from usage.experiments.train.large_world_obstacle_pilot.preview import write_preview_files


def test_direct_waypoint_actions_detect_an_obstacle() -> None:
    class World:
        @staticmethod
        def world_occupied(position):
            return position == (2, 1, 1)

    assert not direct_path_is_free(World(), (1, 1, 1), (3, 1, 1))
    assert direct_path_is_free(World(), (1, 2, 1), (3, 2, 1))


def test_cross_section_walls_have_exactly_one_shrinking_aperture() -> None:
    assert [side for _, side in PORTAL_WALLS] == [32, 16, 8, 4, 2, 1]
    for x, side in PORTAL_WALLS:
        boxes = wall_sources(x, side)
        occupied = sum(
            (box.maximum_inclusive[1] - box.minimum[1] + 1)
            * (box.maximum_inclusive[2] - box.minimum[2] + 1)
            for box in boxes
        )
        assert occupied == EXTENT[1] * EXTENT[2] - side * side

        def blocked(y: int, z: int) -> bool:
            return any(
                box.minimum[1] <= y <= box.maximum_inclusive[1]
                and box.minimum[2] <= z <= box.maximum_inclusive[2]
                for box in boxes
            )

        y0 = PORTAL_CENTER[0] - side // 2
        z0 = PORTAL_CENTER[1] - side // 2
        assert all(not blocked(y, z) for y in range(y0, y0 + side) for z in range(z0, z0 + side))
        assert all(blocked(y0 - 1, z) for z in range(z0, z0 + side))
        assert all(blocked(y0 + side, z) for z in range(z0, z0 + side))
        assert all(blocked(y, z0 - 1) for y in range(y0, y0 + side))
        assert all(blocked(y, z0 + side) for y in range(y0, y0 + side))


def test_compiled_obstacle_routes_require_planning(tmp_path) -> None:
    report = preflight(tmp_path / "worlds", samples_per_stage=1, wall_indices=(0,))

    assert report["extent"] == EXTENT
    assert len(report["sources"]) == len(SOURCES) == 4 * len(PORTAL_WALLS)
    assert {source.minimum[0] for source in SOURCES} == {x for x, _ in PORTAL_WALLS}
    assert report["occupied_voxels"] == sum(
        EXTENT[1] * EXTENT[2] - side * side for _, side in PORTAL_WALLS
    )
    assert report["logical_cells"] == 4_294_967_296
    assert max(source.maximum_inclusive[0] for source in SOURCES) == 3712
    assert max(source.maximum_inclusive[1] for source in SOURCES) > 1900
    assert max(source.maximum_inclusive[2] for source in SOURCES) > 400
    assert len(report["routes"]) == 11
    assert any(route["astar_feasible"] for route in report["routes"])
    assert any(
        route["astar_feasible"] and not route["direct_path_free"]
        for route in report["routes"]
    )
    assert report["portal_walls"][-1] == {"x": 3712, "side": 1, "center": PORTAL_CENTER}
    assert len(report["portal_crossings"]) == len(PORTAL_WALLS)
    assert report["portal_crossings"][-1]["crossing"] == (3712, 1024, 256)
    env = VoxelEnv({
        "agent_count": 1,
        "max_steps": 1,
        "trail_mode": False,
        "extent": EXTENT,
        "compiled_world_path": report["pack_path"],
        "obs_mode": "box",
        "box_radius": 1,
        "action_mode": "discrete_18",
        "waypoints": {"start": (2048, 2000, 500), "goal": (2049, 2000, 500)},
    })
    try:
        env.reset(seed=409)
        world = env._rust_env
        planner = VoxelAStarOracle(env)
        assert not world.world_occupied((144, 176, 128))
        assert not world.world_occupied((2040, 1020, 254))
        for x, side in PORTAL_WALLS:
            y0 = PORTAL_CENTER[0] - side // 2
            z0 = PORTAL_CENTER[1] - side // 2
            assert not world.world_occupied((x, y0, z0))
            assert world.world_occupied((x, y0 - 1, z0))
            assert world.world_occupied((x, y0, z0 - 1))
            crossing = planner._find_path(
                (x - 2, PORTAL_CENTER[0], PORTAL_CENTER[1]),
                (x + 2, PORTAL_CENTER[0], PORTAL_CENTER[1]),
            )
            wall_step = next(point for point in crossing if point[0] == x)
            assert y0 <= wall_step[1] < y0 + side
            assert z0 <= wall_step[2] < z0 + side
            if side == 1:
                assert wall_step == (x, *PORTAL_CENTER)
        assert sum(
            not world.world_occupied((3712, y, z))
            for y in range(1022, 1027)
            for z in range(254, 259)
        ) == 1
    finally:
        env.close()
    previews = write_preview_files(report, tmp_path / "previews")
    assert len(previews) == len(PORTAL_WALLS)
    preview = json.loads(previews[0].read_text(encoding="utf-8"))
    assert preview["world"]["identity_sha256"] == report["world_identity"]
    assert preview["episode"]["start_pos"] == [508, 1024, 256]
    assert (previews[0].parent / preview["world"]["manifest_path"]).resolve().is_file()
    final_portal = json.loads(previews[-1].read_text(encoding="utf-8"))
    assert final_portal["episode"]["start_pos"] == [3708, 1024, 256]
