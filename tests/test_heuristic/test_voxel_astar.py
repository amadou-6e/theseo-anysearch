"""Integration checks for the NetworkX voxel A* oracle."""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import pytest

from theseo_anysearch.experiments.output import OutputStore
from theseo_anysearch.experiments.trajectory import (
    collect_heuristic_episode,
    write_heuristic_trajectory,
)
from tests.test_environments.test_integration._voxel_validity_support import (
    make_radial_test_env,
)
from theseo_anysearch.heuristic.voxel_astar import (
    VoxelAStarOracle,
    VoxelDijkstraHeuristic,
    VoxelReplanningAStarHeuristic,
    VoxelWeightedAStarHeuristic,
)


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

def test_dijkstra_matches_astar_shortest_path_length(tmp_path):
    env = make_radial_test_env(
        tmp_path,
        start=(4, 4, 4),
        goal=(4, 4, 8),
        geometry_boxes=[[4, 4, 6, 4, 4, 6]],
    )
    env.reset(seed=0)

    astar_plan = VoxelAStarOracle(env).plan()
    dijkstra = VoxelDijkstraHeuristic(env)
    dijkstra_plan = dijkstra.plan()
    replay = dijkstra.replay(dijkstra_plan)

    assert dijkstra_plan.steps == astar_plan.steps
    assert replay.goal_reached is True
    assert replay.mismatch is None


def test_weighted_astar_reaches_goal(tmp_path):
    env = make_radial_test_env(
        tmp_path,
        start=(4, 4, 4),
        goal=(4, 4, 8),
        geometry_boxes=[[4, 4, 6, 4, 4, 6]],
    )
    env.reset(seed=0)

    heuristic = VoxelWeightedAStarHeuristic(env, heuristic_weight=2.0)
    plan = heuristic.plan()
    replay = heuristic.replay(plan)

    assert heuristic.heuristic_weight == 2.0
    assert replay.goal_reached is True
    assert replay.mismatch is None


def test_weighted_astar_rejects_non_positive_weight(tmp_path):
    env = make_radial_test_env(tmp_path)

    with pytest.raises(ValueError, match="greater than zero"):
        VoxelWeightedAStarHeuristic(env, heuristic_weight=0.0)


def test_replanning_astar_plans_before_every_step(tmp_path):
    env = make_radial_test_env(
        tmp_path,
        start=(4, 4, 4),
        goal=(4, 4, 8),
        geometry_boxes=[[4, 4, 6, 4, 4, 6]],
        reward_overrides={"trail_mode": True},
    )
    env.reset(seed=0)
    heuristic = VoxelReplanningAStarHeuristic(env)

    with patch.object(heuristic, "plan", wraps=heuristic.plan) as plan_spy:
        replay = heuristic.replay()

    assert replay.goal_reached is True
    assert replay.mismatch is None
    assert plan_spy.call_count == replay.steps_executed

def test_yaml_style_config_collects_dijkstra_reference_episode(tmp_path):
    env = make_radial_test_env(
        tmp_path,
        start=(4, 4, 4),
        goal=(4, 4, 8),
        geometry_boxes=[[4, 4, 6, 4, 4, 6]],
    )
    env_config = dict(env._config)

    episode = collect_heuristic_episode(
        env_config,
        "dijkstra",
        env=env,
        seed=0,
    )

    assert episode.success is True
    assert len(episode.steps) == 4
    assert episode.goal_pos == (4, 4, 8)


def test_heuristic_trajectory_is_replayer_compatible(tmp_path):
    env = make_radial_test_env(
        tmp_path.joinpath("env"),
        start=(4, 4, 4),
        goal=(5, 5, 5),
    )
    episode = collect_heuristic_episode(
        dict(env._config),
        "weighted_astar",
        weight=2.0,
        env=env,
        seed=0,
    )
    store = OutputStore(tmp_path.joinpath("run"))

    relative_path = write_heuristic_trajectory(
        store,
        episode,
        heuristic_type="weighted_astar",
        weight=2.0,
        iteration=40,
        experiment_name="heuristic-test",
        run_id="run-1",
    )
    payload = store.read_json(relative_path)

    assert relative_path == "trajectories/heuristic_weighted_astar.json"
    assert payload["heuristic"] == {"type": "weighted_astar", "weight": 2.0}
    assert payload["episode"]["success"] is True
    assert payload["episode"]["steps_taken"] == 1
