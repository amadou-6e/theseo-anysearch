"""Helpers for deterministic voxel environment validity tests.

Examples
--------
Create a radial test environment with fixed waypoints.

>>> env = make_radial_test_env(tmp_path)
>>> obs, _ = env.reset(seed=0)
>>> set(obs) >= {"goal_distance", "goal_direction", "ray_hits"}
True
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv


GRID_SIZE = 32
MAX_STEPS = 100
STEP_COST = -0.05
GOAL_REWARD = 10.0
DISTANCE_SHAPING = 0.02
COLLISION_COST = -0.5

START = (4, 4, 4)
GOAL = (4, 4, 6)

ACTION_MINUS_X = 4
ACTION_MINUS_Y = 10
ACTION_MINUS_Z = 12
ACTION_PLUS_Z = 13
ACTION_PLUS_Y = 15
ACTION_PLUS_X = 21

RAY_INDEX_PLUS_X = 21
RAY_INDEX_MINUS_X = 4
RAY_TYPE_INDEX_PLUS_Z = 13

MAX_RAY_HIT_TYPE = 5.0
BLOCK_KIND_OCCUPIED = 1.0 / MAX_RAY_HIT_TYPE
BLOCK_KIND_START = 2.0 / MAX_RAY_HIT_TYPE
BLOCK_KIND_GOAL = 3.0 / MAX_RAY_HIT_TYPE
BLOCK_KIND_BOUNDARY = 4.0 / MAX_RAY_HIT_TYPE
BLOCK_KIND_FILLED = 5.0 / MAX_RAY_HIT_TYPE


def write_waypoints_file(tmp_path: Path, start=START, goal=GOAL) -> Path:
    """Write a deterministic waypoint file for the voxel env.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory root.
    start : tuple[int, int, int], default=START
        Fixed start coordinate.
    goal : tuple[int, int, int], default=GOAL
        Fixed goal coordinate.

    Returns
    -------
    Path
        Path to the waypoint JSON file.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path.joinpath("waypoints.json")
    path.write_text(
        json.dumps({"start": list(start), "goal": list(goal)}),
        encoding="utf-8",
    )
    return path


def make_radial_test_env(
    tmp_path: Path,
    *,
    start=START,
    goal=GOAL,
    geometry_boxes: list[list[int]] | None = None,
    reward_overrides: dict | None = None,
) -> VoxelEnv:
    """Create a deterministic single-agent radial voxel environment.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory root.
    start : tuple[int, int, int], default=START
        Fixed start coordinate.
    goal : tuple[int, int, int], default=GOAL
        Fixed goal coordinate.
    geometry_boxes : list[list[int]] | None, optional
        Filled geometry boxes to load into the world.
    reward_overrides : dict | None, optional
        Reward config values that override the default deterministic setup.

    Returns
    -------
    VoxelEnv
        Configured environment matching the current PPO maps reward setup.
    """

    waypoints_file = write_waypoints_file(tmp_path, start=start, goal=goal)
    config = {
        "grid_size": GRID_SIZE,
        "max_steps": MAX_STEPS,
        "seed": 42,
        "obs_mode": "radial",
        "ray_max_len": 16,
        "trail_mode": True,
        "step_cost": STEP_COST,
        "goal_reward": GOAL_REWARD,
        "distance_shaping": DISTANCE_SHAPING,
        "collision_cost": COLLISION_COST,
        "waypoints_file": str(waypoints_file),
        "geometry_boxes": geometry_boxes or [],
    }
    config.update(reward_overrides or {})
    return VoxelEnv(config)


def normalized_cursor(coord: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return normalized cursor coordinates for the configured grid.

    Parameters
    ----------
    coord : tuple[int, int, int]
        Grid coordinate in the env's 1-based coordinate system.

    Returns
    -------
    tuple[float, float, float]
        Normalized cursor position in ``[0, 1]``.
    """

    inv = 1.0 / float(GRID_SIZE - 1)
    return tuple((value - 1) * inv for value in coord)


def normalized_goal_distance(distance: int) -> float:
    """Return the normalized Manhattan goal distance for the configured grid.

    Parameters
    ----------
    distance : int
        Manhattan distance to the goal in grid cells.

    Returns
    -------
    float
        Distance divided by the maximum Manhattan distance.
    """

    return distance / float(3 * (GRID_SIZE - 1))


def normalized_goal_direction(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Return the Euclidean unit direction from start to goal.

    Parameters
    ----------
    start : tuple[int, int, int]
        Current cursor coordinate.
    goal : tuple[int, int, int]
        Goal coordinate.

    Returns
    -------
    tuple[float, float, float]
        Unit-length signed direction, or zeros when start equals goal.
    """

    delta = tuple(goal_value - start_value for start_value, goal_value in zip(start, goal))
    norm = math.sqrt(sum(value * value for value in delta))
    if norm == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(value / norm for value in delta)
