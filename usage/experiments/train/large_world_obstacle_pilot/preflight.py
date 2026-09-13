"""Build and preflight an obstacle-rich compiled world for waypoint training.

The generated pack and JSON report stay under ignored ``runtime/``. The source
boxes and route seeds are fixed here so the preflight is reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from theseo_anysearch.environments.action_spaces import (
    offsets_for_mode,
    shortest_actions,
)
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic.voxel.astar.standard import VoxelAStarOracle
from theseo_anysearch.rllib.trainer.waypoint_routes import route_distance, sample_route
from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent
from usage.experiments.train.large_world_obstacle_pilot.curriculum import (
    MAX_EPISODE_STEPS,
    STAGE_LENGTHS,
    gate_routes,
)

EXTENT = (4096, 2048, 512)
ACTION_MODE = "discrete_18"
ROUTE_LENGTH = 96
MAX_PLANNED_STEPS = 128

# Full YZ cross-section walls divide the long X axis into progressively harder
# chambers. Each wall is one voxel thick and has exactly one square aperture;
# the final aperture is the single voxel (x, 1024, 256).
PORTAL_WALLS = (
    (512, 32),
    (1152, 16),
    (1792, 8),
    (2432, 4),
    (3072, 2),
    (3712, 1),
)
PORTAL_CENTER = (1024, 256)


def portal_bounds(side: int) -> tuple[int, int, int, int]:
    """Inclusive YZ bounds of a centered square aperture."""
    y0 = PORTAL_CENTER[0] - side // 2
    z0 = PORTAL_CENTER[1] - side // 2
    return y0, y0 + side - 1, z0, z0 + side - 1


def wall_sources(x: int, side: int) -> tuple[BoxSource, ...]:
    """Tile a whole YZ plane with four boxes, leaving one exact aperture."""
    y0, y1, z0, z1 = portal_bounds(side)
    return (
        BoxSource((x, 0, 0), (x, y0 - 1, EXTENT[2] - 1)),
        BoxSource((x, y1 + 1, 0), (x, EXTENT[1] - 1, EXTENT[2] - 1)),
        BoxSource((x, y0, 0), (x, y1, z0 - 1)),
        BoxSource((x, y0, z1 + 1), (x, y1, EXTENT[2] - 1)),
    )


WALL_SOURCES = tuple(
    source for x, side in PORTAL_WALLS for source in wall_sources(x, side)
)
SOURCES = WALL_SOURCES
WALL_INDICES = tuple(range(len(PORTAL_WALLS)))


def route_start(wall_index: int) -> tuple[int, int, int]:
    x, _ = PORTAL_WALLS[wall_index]
    return x - 4, *PORTAL_CENTER


def direct_path_is_free(world: object, start: tuple[int, int, int], goal: tuple[int, int, int]) -> bool:
    """Check the exact actions the current continue-route collector would use."""
    position = start
    offsets = offsets_for_mode(ACTION_MODE)
    for action in shortest_actions(start, goal, ACTION_MODE):
        offset = offsets[int(action)]
        position = tuple(a + b for a, b in zip(position, offset))
        if world.world_occupied(position):
            return False
    return position == goal


def replay_astar_route(pack_path: Path, route: object, seed: int) -> dict:
    """Execute one planned detour in the actual compiled-world environment."""
    env = VoxelEnv(
        {
            "agent_count": 1,
            "max_steps": MAX_PLANNED_STEPS,
            "trail_mode": False,
            "extent": EXTENT,
            "compiled_world_path": str(pack_path),
            "obs_mode": "box",
            "box_radius": 1,
            "action_mode": ACTION_MODE,
            "waypoint_route": route.model_dump(mode="python"),
        }
    )
    try:
        env.reset(seed=seed)
        planner = VoxelAStarOracle(env)
        steps = 0
        for _ in route.waypoints:
            plan = planner.plan()
            if not plan.action_indices:
                return {"seed": seed, "success": False, "steps": steps, "failure": "empty_plan"}
            for action in plan.action_indices:
                _, _, terminated, truncated, info = env.step(action)
                steps += 1
                if info["collision"] or truncated:
                    return {"seed": seed, "success": False, "steps": steps, "failure": "collision_or_budget"}
                if info["waypoint_reached"]:
                    if info["goal_reached"]:
                        return {"seed": seed, "success": True, "steps": steps, "failure": None}
                    break
                if terminated:
                    return {"seed": seed, "success": False, "steps": steps, "failure": "early_termination"}
            else:
                return {"seed": seed, "success": False, "steps": steps, "failure": "plan_not_reached"}
        return {"seed": seed, "success": False, "steps": steps, "failure": "route_not_complete"}
    finally:
        env.close()


def preflight(
    cache_dir: Path,
    samples_per_stage: int,
    wall_indices: tuple[int, ...] = WALL_INDICES,
) -> dict:
    """Compare direct route actions with obstacle-aware A* on fixed random routes."""
    # The route starts and goals are fixed by this preflight. The generic
    # surface-candidate index would enumerate millions of full-wall voxels,
    # consume excessive memory, and is not used by these waypoint routes.
    compiled = compile_world(
        SOURCES, WorldExtent.from_value(EXTENT), cache_dir,
        generate_candidates=False,
    )
    env = VoxelEnv(
        {
            "agent_count": 1,
            "max_steps": MAX_PLANNED_STEPS,
            "trail_mode": False,
            "extent": EXTENT,
            "compiled_world_path": str(compiled.root),
            "obs_mode": "box",
            "box_radius": 1,
            "action_mode": ACTION_MODE,
            # Keep the runtime cursor and goal out of sampled routes: both are
            # represented as occupied overlay cells in world_occupied().
            "waypoints": {"start": (2048, 2000, 500), "goal": (2049, 2000, 500)},
        }
    )
    try:
        env.reset(seed=409)
        world = env._rust_env
        planner = VoxelAStarOracle(env)
        curriculum_stages = []
        for stage, (length, route) in enumerate(zip(STAGE_LENGTHS, gate_routes())):
            points = (route.start, *route.waypoints)
            if route_distance(route, ACTION_MODE) != length:
                raise ValueError("gate curriculum route has the wrong action length")
            if any(world.world_occupied(point) for point in points) or not all(
                direct_path_is_free(world, start, goal)
                for start, goal in zip(points, points[1:])
            ):
                raise ValueError("gate curriculum route crosses occupied geometry")
            curriculum_stages.append({
                "stage": stage,
                "route_length": length,
                "route": route.model_dump(mode="python"),
                "direct_path_free": True,
            })
        results = []
        replay = None
        for wall_index in wall_indices:
            for stage in range(11):
                distance = min(1 + 2 * stage, 20)
                for sample in range(samples_per_stage):
                    seed = 409_000 + wall_index * 100_000 + stage * 1_000 + sample
                    route = sample_route(
                        start=route_start(wall_index),
                        total_distance=ROUTE_LENGTH,
                        segment_distance=distance,
                        action_mode=ACTION_MODE,
                        seed=seed,
                        extent=EXTENT,
                    )
                    points = (route.start, *route.waypoints)
                    direct_free = True
                    astar_feasible = True
                    astar_steps = 0
                    expanded_nodes = 0
                    failure = None
                    for start, goal in zip(points, points[1:]):
                        if world.world_occupied(start) or world.world_occupied(goal):
                            astar_feasible = False
                            direct_free = False
                            failure = "occupied_endpoint"
                            break
                        direct_free &= direct_path_is_free(world, start, goal)
                        try:
                            path = planner._find_path(start, goal)
                        except nx.NetworkXNoPath:
                            astar_feasible = False
                            failure = "no_path"
                            break
                        astar_steps += len(path) - 1
                        expanded_nodes += planner._last_search_nodes
                    if astar_feasible and astar_steps > MAX_PLANNED_STEPS:
                        failure = "episode_budget"
                    results.append(
                        {
                            "wall_index": wall_index,
                            "start": route.start,
                            "stage": stage,
                            "seed": seed,
                            "segment_distance": distance,
                            "waypoint_count": len(route.waypoints),
                            "direct_path_free": direct_free,
                            "astar_feasible": astar_feasible,
                            "astar_steps": astar_steps if astar_feasible else None,
                            "within_episode_budget": astar_feasible and astar_steps <= MAX_PLANNED_STEPS,
                            "expanded_nodes": expanded_nodes,
                            "failure": failure,
                        }
                    )
                    if replay is None and astar_feasible and not direct_free and astar_steps <= MAX_PLANNED_STEPS:
                        replay = replay_astar_route(compiled.root, route, seed)
        portal_crossings = []
        for x, side in PORTAL_WALLS:
            start = (x - 2, *PORTAL_CENTER)
            goal = (x + 2, *PORTAL_CENTER)
            path = planner._find_path(start, goal)
            wall_step = next(point for point in path if point[0] == x)
            portal_crossings.append({
                "x": x,
                "side": side,
                "crossing": wall_step,
                "steps": len(path) - 1,
            })
        return {
            "world_identity": compiled.manifest.identity_sha256,
            "pack_path": str(compiled.root),
            "extent": EXTENT,
            "logical_cells": EXTENT[0] * EXTENT[1] * EXTENT[2],
            "candidate_index": "empty",
            "occupied_voxels": sum(chunk.occupied_voxels for chunk in compiled.manifest.chunks),
            "route_walls": [
                {"index": index, "start": route_start(index)} for index in wall_indices
            ],
            "portal_walls": [
                {"x": x, "side": side, "center": PORTAL_CENTER}
                for x, side in PORTAL_WALLS
            ],
            "sources": [
                {"minimum": source.minimum, "maximum_inclusive": source.maximum_inclusive}
                for source in SOURCES
            ],
            "action_mode": ACTION_MODE,
            "route_length": ROUTE_LENGTH,
            "episode_budget": MAX_PLANNED_STEPS,
            "curriculum_max_steps": MAX_EPISODE_STEPS,
            "curriculum_stages": curriculum_stages,
            "routes": results,
            "portal_crossings": portal_crossings,
            "astar_detour_replay": replay,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/obstacle-waypoint-pilot"))
    parser.add_argument("--samples-per-stage", type=int, default=2)
    args = parser.parse_args()
    if args.samples_per_stage < 1:
        parser.error("--samples-per-stage must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(args.output_dir / "worlds", args.samples_per_stage)
    path = args.output_dir / "preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    routes = report["routes"]
    print(json.dumps({
        "report": str(path),
        "world_identity": report["world_identity"],
        "curriculum_lengths": [
            stage["route_length"] for stage in report["curriculum_stages"]
        ],
        "curriculum_max_steps": report["curriculum_max_steps"],
        "routes": len(routes),
        "astar_feasible": sum(item["astar_feasible"] for item in routes),
        "within_episode_budget": sum(item["within_episode_budget"] for item in routes),
        "direct_path_free": sum(item["direct_path_free"] for item in routes),
        "astar_detour_replay": report["astar_detour_replay"],
    }, indent=2))


if __name__ == "__main__":
    main()
