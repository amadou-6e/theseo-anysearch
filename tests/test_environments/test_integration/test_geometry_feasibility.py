"""Feasibility gates for randomly augmented geometry-pool episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv


def _pool(tmp_path: Path, grids: list[np.ndarray]) -> Path:
    root = tmp_path.joinpath("pool")
    root.mkdir()
    root.joinpath("pool_meta.json").write_text(
        json.dumps({"grid_size": int(grids[0].shape[0])}), encoding="utf-8"
    )
    for index, grid in enumerate(grids):
        np.save(root.joinpath(f"{index:04d}.npy"), grid.astype(np.uint8))
    return root


def _config(
    pool_dir: Path,
    *,
    start=(2, 2, 2),
    goal=(4, 2, 2),
    action_mode="discrete_26",
    max_steps=20,
    maximum_attempts=1,
    maximum_search_nodes=1_000,
    recovery_margin_steps=0,
) -> dict:
    return {
        "grid_size": 5,
        "seed": 7,
        "max_steps": max_steps,
        "trail_mode": False,
        "action_mode": action_mode,
        "waypoints": {"start": start, "goal": goal},
        "geometry_pool": {
            "pool_dir": str(pool_dir),
            "augmentation": {
                "feasibility": {
                    "maximum_attempts": maximum_attempts,
                    "maximum_search_nodes": maximum_search_nodes,
                    "recovery_margin_steps": recovery_margin_steps,
                }
            },
        },
    }


@pytest.mark.parametrize(
    ("occupied", "reason"),
    [((2, 2, 2), "occupied_start"), ((4, 2, 2), "occupied_goal")],
)
def test_rejects_occupied_endpoints(tmp_path, occupied, reason):
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    grid[tuple(axis - 1 for axis in occupied)] = 1
    env = VoxelEnv(_config(_pool(tmp_path, [grid])))

    with pytest.raises(RuntimeError, match=reason):
        env.reset(seed=11)


def test_rejects_disconnected_task(tmp_path):
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    grid[2, :, :] = 1
    env = VoxelEnv(
        _config(_pool(tmp_path, [grid]), start=(2, 2, 2), goal=(4, 2, 2))
    )

    with pytest.raises(RuntimeError, match="no_path"):
        env.reset(seed=11)


def test_exact_action_mode_and_episode_budget_are_enforced(tmp_path):
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    pool = _pool(tmp_path, [grid])
    diagonal = dict(start=(2, 2, 2), goal=(3, 3, 2), max_steps=1)

    _, info = VoxelEnv(
        _config(pool, action_mode="discrete_26", **diagonal)
    ).reset(seed=11)
    assert info["geometry_feasibility"]["accepted_plan_steps"] == 1

    with pytest.raises(RuntimeError, match="episode_budget_exceeded"):
        VoxelEnv(_config(pool, action_mode="discrete_6", **diagonal)).reset(seed=11)


def test_recovery_margin_rejects_detour_that_fits_without_margin(tmp_path):
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    grid[2, 1, 1] = 1
    env = VoxelEnv(
        _config(
            _pool(tmp_path, [grid]),
            start=(2, 2, 2),
            goal=(4, 2, 2),
            max_steps=2,
            recovery_margin_steps=1,
        )
    )

    with pytest.raises(RuntimeError, match="episode_budget_exceeded"):
        env.reset(seed=11)


def test_planner_budget_exhaustion_is_categorized(tmp_path):
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    env = VoxelEnv(
        _config(
            _pool(tmp_path, [grid]),
            start=(1, 1, 1),
            goal=(5, 5, 5),
            maximum_search_nodes=1,
        )
    )

    with pytest.raises(RuntimeError, match="planner_budget_exhausted"):
        env.reset(seed=11)


def test_bounded_resampling_is_deterministic_and_reports_rejections(tmp_path):
    blocked = np.zeros((5, 5, 5), dtype=np.uint8)
    blocked[1, 1, 1] = 1
    clear = np.zeros((5, 5, 5), dtype=np.uint8)
    pool = _pool(tmp_path, [blocked, clear])

    def run_once():
        env = VoxelEnv(_config(pool, maximum_attempts=2))
        samples = iter((blocked, clear))
        env._geo_pool.sample = lambda rng=None: next(samples)
        _, info = env.reset(seed=11)
        return info["geometry_feasibility"], tuple(env._scenario_geometry)

    first = run_once()
    second = run_once()
    assert first == second
    assert first[0] == {
        "enabled": True,
        "attempts": 2,
        "rejections": {"occupied_start": 1},
        "accepted_plan_steps": 2,
    }


def test_explicit_reset_seed_replays_the_same_sampling_sequence(tmp_path):
    first_grid = np.zeros((5, 5, 5), dtype=np.uint8)
    first_grid[0, 4, 4] = 1
    second_grid = np.zeros((5, 5, 5), dtype=np.uint8)
    second_grid[4, 0, 4] = 1
    env = VoxelEnv(_config(_pool(tmp_path, [first_grid, second_grid])))

    _, first_info = env.reset(seed=91)
    first_geometry = tuple(env._scenario_geometry)
    _, second_info = env.reset(seed=91)

    assert tuple(env._scenario_geometry) == first_geometry
    assert second_info["geometry_feasibility"] == first_info["geometry_feasibility"]


def test_curriculum_selected_segment_is_validated_after_selection(tmp_path):
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    grid[3, 2, 2] = 1
    env = VoxelEnv(_config(_pool(tmp_path, [grid])))
    env.set_waypoint_curriculum(
        [((2, 3, 3), (4, 3, 3))],
        [1.0],
    )

    with pytest.raises(RuntimeError, match="occupied_goal"):
        env.reset(seed=11)


def test_disabled_gate_preserves_impossible_geometry(tmp_path):
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    grid[1, 1, 1] = 1
    config = _config(_pool(tmp_path, [grid]))
    config["geometry_pool"]["augmentation"]["feasibility"]["enabled"] = False

    _, info = VoxelEnv(config).reset(seed=11)

    assert "geometry_feasibility" not in info


def test_shared_validator_rejects_fixed_box_geometry() -> None:
    config = {
        "grid_size": 5,
        "max_steps": 20,
        "trail_mode": False,
        "waypoints": {"start": (2, 2, 2), "goal": (4, 2, 2)},
        "geometry_boxes": [[4, 2, 2, 4, 2, 2]],
        "geometry_validation": {
            "enabled": True,
            "maximum_attempts": 1,
            "maximum_search_nodes": 1_000,
            "recovery_margin_steps": 0,
        },
    }

    with pytest.raises(RuntimeError, match="occupied_goal"):
        VoxelEnv(config).reset(seed=11)


def test_shared_validator_rejects_fixed_stl_geometry() -> None:
    config = {
        "grid_size": 5,
        "max_steps": 20,
        "trail_mode": False,
        "stl_path": "synthetic.stl",
        "waypoints": {"start": (2, 2, 2), "goal": (4, 2, 2)},
        "geometry_validation": {
            "enabled": True,
            "maximum_attempts": 1,
            "maximum_search_nodes": 1_000,
            "recovery_margin_steps": 0,
        },
    }

    with patch(
        "theseo_anysearch.environments.pettingzoo.multi_voxel_env._load_stl_geometry",
        return_value=[(4, 2, 2)],
    ):
        with pytest.raises(RuntimeError, match="occupied_goal"):
            VoxelEnv(config).reset(seed=11)
