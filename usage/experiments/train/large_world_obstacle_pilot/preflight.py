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
from theseo_anysearch.rllib.trainer.waypoint_routes import sample_route
from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent

EXTENT = (128, 96, 64)
START = (16, 48, 32)
ACTION_MODE = "discrete_18"
ROUTE_LENGTH = 96
MAX_PLANNED_STEPS = 128

# Three two-voxel-thick partitions. Each has a different 16 x 12 doorway,
# forcing some shortest paths to detour while keeping the chambers connected.
SOURCES = (
    BoxSource((31, 0, 0), (32, 39, 63)),
    BoxSource((31, 56, 0), (32, 95, 63)),
    BoxSource((31, 40, 0), (32, 55, 25)),
    BoxSource((31, 40, 38), (32, 55, 63)),
    BoxSource((63, 0, 0), (64, 19, 63)),
    BoxSource((63, 36, 0), (64, 95, 63)),
    BoxSource((63, 20, 0), (64, 35, 19)),
    BoxSource((63, 20, 32), (64, 35, 63)),
    BoxSource((95, 0, 0), (96, 54, 63)),
    BoxSource((95, 71, 0), (96, 95, 63)),
    BoxSource((95, 55, 0), (96, 70, 27)),
    BoxSource((95, 55, 40), (96, 70, 63)),
    BoxSource((43, 12, 8), (49, 19, 42)),
    BoxSource((76, 68, 14), (83, 78, 48)),
    BoxSource((105, 27, 6), (111, 37, 44)),
)


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


def preflight(cache_dir: Path, samples_per_stage: int) -> dict:
    """Compare direct route actions with obstacle-aware A* on fixed random routes."""
    compiled = compile_world(SOURCES, WorldExtent.from_value(EXTENT), cache_dir)
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
            "waypoints": {"start": (127, 95, 63), "goal": (128, 95, 63)},
        }
    )
    try:
        env.reset(seed=409)
        world = env._rust_env
        planner = VoxelAStarOracle(env)
        results = []
        replay = None
        for stage in range(11):
            distance = min(1 + 2 * stage, 20)
            for sample in range(samples_per_stage):
                seed = 409_000 + stage * 1_000 + sample
                route = sample_route(
                    start=START,
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
        return {
            "world_identity": compiled.manifest.identity_sha256,
            "pack_path": str(compiled.root),
            "extent": EXTENT,
            "occupied_voxels": sum(chunk.occupied_voxels for chunk in compiled.manifest.chunks),
            "sources": [
                {"minimum": source.minimum, "maximum_inclusive": source.maximum_inclusive}
                for source in SOURCES
            ],
            "action_mode": ACTION_MODE,
            "route_length": ROUTE_LENGTH,
            "episode_budget": MAX_PLANNED_STEPS,
            "routes": results,
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
        "routes": len(routes),
        "astar_feasible": sum(item["astar_feasible"] for item in routes),
        "within_episode_budget": sum(item["within_episode_budget"] for item in routes),
        "direct_path_free": sum(item["direct_path_free"] for item in routes),
        "astar_detour_replay": report["astar_detour_replay"],
    }, indent=2))


if __name__ == "__main__":
    main()
