"""Integration checks for the NetworkX voxel A* oracle."""

from __future__ import annotations

import networkx as nx
import pytest

from tests.test_environments.test_integration._voxel_validity_support import (
    make_radial_test_env,
)
from theseo_anysearch.heuristic.voxel_astar import VoxelAStarOracle


@pytest.mark.parametrize("trail_mode", [False, True])
def test_astar_plan_replays_to_goal(tmp_path, trail_mode):
    env = make_radial_test_env(
        tmp_path,
        start=(4, 4, 4),
        goal=(4, 4, 8),
        geometry_boxes=[[4, 4, 6, 4, 4, 6]],
        reward_overrides={"trail_mode": trail_mode},
    )
    env.reset(seed=0)

    plan = VoxelAStarOracle(env).plan()
    replay = VoxelAStarOracle(env).replay(plan)

    assert plan.positions[0] == (4, 4, 4)
    assert plan.positions[-1] == (4, 4, 8)
    assert (4, 4, 6) not in plan.positions
    assert plan.steps == 4
    assert replay.goal_reached is True
    assert replay.terminated is True
    assert replay.mismatch is None
    assert replay.positions == plan.positions


def test_astar_reports_unsolvable_geometry(tmp_path):
    env = make_radial_test_env(
        tmp_path,
        start=(1, 1, 1),
        goal=(3, 3, 3),
        geometry_boxes=[[1, 1, 1, 2, 2, 2]],
    )
    env.reset(seed=0)

    with pytest.raises(nx.NetworkXNoPath):
        VoxelAStarOracle(env).plan()
