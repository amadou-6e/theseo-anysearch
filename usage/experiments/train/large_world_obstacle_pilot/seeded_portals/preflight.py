"""Validate seed-selected gate packs with native occupancy and A* execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic.voxel.astar.standard import VoxelAStarOracle
from theseo_anysearch.rllib.trainer.waypoint_routes import route_distance
from theseo_anysearch.worlds.seeded_catalog import load_catalog
from usage.experiments.train.large_world_obstacle_pilot.curriculum import (
    MAX_EPISODE_STEPS, STAGE_LENGTHS,
)
from usage.experiments.train.large_world_obstacle_pilot.preflight import (
    ACTION_MODE, PORTAL_WALLS, portal_bounds,
)


def check_episode(catalog_path: Path, stage: int, seed: int) -> dict:
    catalog = load_catalog(catalog_path)
    variant = catalog.for_seed(seed)
    route = catalog.route_for_stage(stage, seed, action_mode=ACTION_MODE)
    expected = STAGE_LENGTHS[stage]
    if route_distance(route, ACTION_MODE) != expected:
        raise ValueError("seeded stage route has wrong action length")
    env = VoxelEnv({
        "agent_count": 1, "max_steps": MAX_EPISODE_STEPS, "trail_mode": False,
        "extent": catalog.extent, "compiled_world_catalog_path": str(catalog_path),
        "obs_mode": "box", "box_radius": 1, "action_mode": ACTION_MODE,
        "waypoint_route": route.model_dump(mode="python"),
    })
    try:
        # world_occupied includes the live goal overlay. Probe with an unrelated
        # two-step goal before installing this route, or its first waypoint
        # would appear occupied even in an empty compiled cell.
        probe_route = {"start": (1, 1024, 256), "waypoints": [(3, 1024, 256)]}
        env.queue_waypoint_route(probe_route["start"], probe_route["waypoints"])
        env.reset(seed=seed)
        if any(env._rust_env.world_occupied(point) for point in route.waypoints):
            raise ValueError("seeded route contains an occupied waypoint")
        _, reset_info = env.reset(seed=seed)
        if reset_info["world_identity_sha256"] != variant.identity_sha256:
            raise ValueError("reset selected the wrong seed-dependent world")
        for (x, side), center in zip(PORTAL_WALLS, variant.portal_centers):
            y0, y1, z0, z1 = portal_bounds(side, center)
            if env._rust_env.world_occupied((x, *center)):
                raise ValueError("portal center is occupied")
            for outside in ((x, y0 - 1, center[1]), (x, y1 + 1, center[1]),
                            (x, center[0], z0 - 1), (x, center[0], z1 + 1)):
                if not env._rust_env.world_occupied(outside):
                    raise ValueError("portal wall has an extra opening")
        planner = VoxelAStarOracle(env)
        steps = 0
        lateral = 0
        crossings: list[tuple[int, int, int]] = []
        for _ in route.waypoints:
            plan = planner.plan()
            if not plan.action_indices:
                raise ValueError("A* returned an empty plan")
            for action in plan.action_indices:
                before = tuple(env._rust_env.cursor_pos())
                _, _, terminated, truncated, info = env.step(action)
                after = tuple(env._rust_env.cursor_pos())
                steps += 1
                lateral += int(before[1:] != after[1:])
                if after[0] in {x for x, _ in PORTAL_WALLS} and after[0] != before[0]:
                    crossings.append(after)
                if info["collision"] or truncated:
                    raise ValueError("A* route collided or exceeded the episode budget")
                if info["waypoint_reached"]:
                    break
                if terminated:
                    raise ValueError("A* route terminated before its goal")
            else:
                raise ValueError("A* plan did not reach its waypoint")
        if not info["goal_reached"] or steps != expected:
            raise ValueError(f"executed {steps} actions, expected {expected}")
        crossed = [point for point in crossings if point[0] <= route.goal[0]]
        if stage == 11 and (len(crossed) != len(PORTAL_WALLS)
                            or lateral < 600):
            raise ValueError("final route did not traverse six staggered portals")
        return {
            "seed": seed, "stage": stage, "world_identity_sha256": variant.identity_sha256,
            "actions": steps, "lateral_actions": lateral,
            "straight_x_fraction": round((steps - lateral) / steps, 6),
            "wall_crossings": [list(point) for point in crossed],
            "success": True,
        }
    finally:
        env.close()


def preflight(catalog_path: Path, seeds: list[int]) -> dict:
    catalog = load_catalog(catalog_path)
    results = [check_episode(catalog_path, stage, seed)
               for seed in seeds for stage in (8, 9, 10, 11)]
    if len({tuple(tuple(point) for point in result["wall_crossings"])
            for result in results if result["stage"] == 11}) < 2:
        raise ValueError("sampled seeds did not produce distinct portal traversals")
    return {"catalog_sha256": catalog.identity_sha256, "seeds": seeds,
            "checked_episodes": len(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--report", type=Path, default=Path("runtime/seeded-gate-pilot/preflight.json"))
    args = parser.parse_args()
    report = preflight(args.catalog, args.seeds)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog_sha256": report["catalog_sha256"],
                      "checked_episodes": report["checked_episodes"],
                      "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
