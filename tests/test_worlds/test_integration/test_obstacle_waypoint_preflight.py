"""Obstacle-world waypoint preflight exercises the compiled regional backend."""

import json
from pathlib import Path

from theseo_anysearch.environments.action_spaces import shortest_actions
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.experiments.loader import load_experiment
from theseo_anysearch.heuristic.voxel.astar.standard import VoxelAStarOracle
from theseo_anysearch.imitation.dataset import collect_demonstrations
from theseo_anysearch.imitation.models import ImitationConfig
from theseo_anysearch.models import WaypointCurriculumConfig
from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
    WaypointCurriculum,
    configure_initial_waypoint_curriculum,
)
from theseo_anysearch.rllib.trainer.waypoint_routes import route_distance
from theseo_anysearch.settings.environment import EnvConfig
from theseo_anysearch.worlds import world_contract
from usage.experiments.train.large_world_obstacle_pilot.curriculum import (
    STAGE_LENGTHS,
    gate_curriculum_settings,
    gate_routes,
)
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


def test_large_world_training_smoke_config_matches_gate_curriculum() -> None:
    config_path = Path(
        "usage/experiments/train/large_world_obstacle_pilot/experiment.yaml"
    )
    config = load_experiment(config_path)
    assert config.training.iterations == 2
    assert config.imitation.generation.episodes == 128
    assert config.imitation.generation.provider.name == "astar"
    assert config.env.waypoint_curriculum.fixed_route_variation_radius == 2
    assert config.env.max_steps == 4608
    assert config.env.to_runtime_dict()["world_identity_sha256"] == (
        "ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05"
    )
    pack_manifest = json.loads(
        (config.env.geometry.compiled_world_path / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert world_contract(config.env.to_runtime_dict())["identity_sha256"] == (
        pack_manifest["identity_sha256"]
    )
    assert config.experiment.output_dir == (
        Path("runtime/gv").resolve()
    )
    assert [
        route.model_dump(mode="python") for route in config.env.waypoint_curriculum.routes
    ] == gate_curriculum_settings()["routes"]


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
    assert [stage["route_length"] for stage in report["curriculum_stages"]] == list(STAGE_LENGTHS)
    assert len(report["curriculum_stages"]) == 12
    assert all(stage["direct_path_free"] for stage in report["curriculum_stages"])
    assert report["curriculum_stages"][-1]["route"]["waypoints"][-1] == (4095, 1026, 256)
    curriculum = WaypointCurriculum(
        WaypointCurriculumConfig.model_validate(gate_curriculum_settings()),
        {"extent": EXTENT, "max_steps": report["curriculum_max_steps"],
         "action_mode": "discrete_18", "compiled_world_path": report["pack_path"]},
    )
    assert curriculum.maximum_stage == 11
    assert curriculum.observe(1, 1)
    assert [
        route_distance(route, "discrete_18")
        for route in curriculum.configured_route_stages({"extent": EXTENT})
    ] == list(STAGE_LENGTHS)
    configured_env = EnvConfig.model_validate({
        "max_steps": report["curriculum_max_steps"],
        "agent_count": 1,
        "trail_mode": False,
        "geometry": {"extent": EXTENT, "compiled_world_path": report["pack_path"]},
        "observation": {"mode": "box", "box_radius": 1},
        "action": {"mode": "discrete_18"},
        "waypoint_curriculum": gate_curriculum_settings(),
    })
    runtime_config = configured_env.to_runtime_dict()
    runtime_config["world_identity_sha256"] = report["world_identity"]
    demonstrations = collect_demonstrations(
        runtime_config,
        ImitationConfig.model_validate({
            "enabled": True,
            "generation": {
                "provider": "astar",
                "episodes": 24,
                "max_attempts": 48,
                "require_success": True,
            },
            "collection": {
                "seed_start": 1000,
                "validation_fraction": 0.1,
                "curriculum_stages": "all",
            },
        }),
    )
    assert demonstrations.manifest.stage_episode_counts == [2] * 12
    assert demonstrations.manifest.attempted_episodes == 24
    assert (
        demonstrations.manifest.training_samples
        + demonstrations.manifest.validation_samples
    ) == 2 * sum(STAGE_LENGTHS)
    assert demonstrations.manifest.world_identity_sha256 == report["world_identity"]
    stage_env = VoxelEnv(runtime_config)
    try:
        configure_initial_waypoint_curriculum(stage_env, runtime_config)
        stage_env.reset(seed=409)
        for action in shortest_actions((1, 1024, 256), (3, 1024, 256), "discrete_18"):
            _, _, terminated, truncated, info = stage_env.step(action)
            assert not truncated
        assert terminated and info["goal_reached"]
    finally:
        stage_env.close()
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
        long_path = planner._find_path((1, *PORTAL_CENTER), (4095, *PORTAL_CENTER))
        assert len(long_path) - 1 == 4094
        assert planner._last_search_nodes == 4095
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
    final_route = gate_routes()[-1]
    route_env = VoxelEnv({
        "agent_count": 1,
        "max_steps": report["curriculum_max_steps"],
        "trail_mode": False,
        "extent": EXTENT,
        "compiled_world_path": report["pack_path"],
        "obs_mode": "box",
        "box_radius": 1,
        "action_mode": "discrete_18",
        "waypoint_route": final_route.model_dump(mode="python"),
    })
    try:
        route_env.reset(seed=409)
        steps = 0
        for start, goal in zip(
            (final_route.start, *final_route.waypoints[:-1]),
            final_route.waypoints,
        ):
            for action in shortest_actions(start, goal, "discrete_18"):
                _, _, terminated, truncated, info = route_env.step(action)
                steps += 1
                assert not info["collision"]
                assert not truncated
        assert steps == 4096
        assert terminated and info["goal_reached"]
    finally:
        route_env.close()
    previews = write_preview_files(report, tmp_path / "previews")
    assert len(previews) == len(PORTAL_WALLS)
    preview = json.loads(previews[0].read_text(encoding="utf-8"))
    assert preview["world"]["identity_sha256"] == report["world_identity"]
    assert preview["episode"]["start_pos"] == [508, 1024, 256]
    assert (previews[0].parent / preview["world"]["manifest_path"]).resolve().is_file()
    final_portal = json.loads(previews[-1].read_text(encoding="utf-8"))
    assert final_portal["episode"]["start_pos"] == [3708, 1024, 256]
