"""Independent geometry-validity and task-feasibility contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from theseo_anysearch.environments.action_spaces import offsets_for_mode
from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv
from theseo_anysearch.environments.validation import (
    BoundedWorldRead,
    InMemoryOccupancy,
    validate_geometry,
    validate_task_feasibility,
)
from theseo_anysearch.settings.environment.geometry import (
    GeometryValidationConfig,
    RoutingDifficultyBand,
)
from theseo_anysearch.experiments.trajectory import _VoxelEpisodeState


def test_geometry_validation_is_independent_of_task_planning() -> None:
    valid = validate_geometry([(1, 1, 1), (3, 3, 3)], (3, 3, 3))
    duplicate = validate_geometry([(1, 1, 1), (1, 1, 1)], (3, 3, 3))
    outside = validate_geometry([(4, 1, 1)], (3, 3, 3))

    assert valid.valid and valid.coordinate_count == 2
    assert duplicate.rejection_reason == "duplicate_coordinate"
    assert outside.rejection_reason == "out_of_bounds"


def test_task_result_exposes_planner_path_length_and_search_counts() -> None:
    result = validate_task_feasibility(
        InMemoryOccupancy([(2, 2, 2)]),
        start=(1, 2, 2),
        goal=(3, 2, 2),
        extent=(4, 4, 4),
        directions=offsets_for_mode("discrete_6"),
        action_mode="discrete_6",
        maximum_search_nodes=100,
        maximum_steps=10,
    )

    assert result.feasible
    assert result.planner == "astar"
    assert result.path[0] == (1, 2, 2)
    assert result.path[-1] == (3, 2, 2)
    assert result.path_length == 4
    assert result.graph_nodes > 0
    assert result.graph_edges > 0


def test_task_rejection_categories_are_stable() -> None:
    occupied = InMemoryOccupancy([(1, 1, 1), (3, 3, 3)])
    common = {
        "extent": (3, 3, 3),
        "directions": offsets_for_mode("discrete_6"),
        "action_mode": "discrete_6",
        "maximum_search_nodes": 100,
        "maximum_steps": 10,
    }

    assert validate_task_feasibility(
        occupied, start=(1, 1, 1), goal=(2, 2, 2), **common
    ).rejection_reason == "occupied_start"
    assert validate_task_feasibility(
        occupied, start=(2, 2, 2), goal=(3, 3, 3), **common
    ).rejection_reason == "occupied_goal"
    assert validate_task_feasibility(
        occupied, start=None, goal=(2, 2, 2), **common
    ).rejection_reason == "missing_goal"


def test_bounded_adapter_uses_only_point_queries() -> None:
    queries: list[tuple[int, int, int]] = []

    def occupied(coordinate):
        queries.append(coordinate)
        return False

    result = validate_task_feasibility(
        BoundedWorldRead(occupied),
        start=(1, 1, 1),
        goal=(2, 1, 1),
        extent=(60_000, 40_000, 20_000),
        directions=offsets_for_mode("discrete_6"),
        action_mode="discrete_6",
        maximum_search_nodes=10,
        maximum_steps=2,
    )

    assert result.feasible
    assert 0 < len(queries) < 20


def test_planner_and_episode_budgets_are_separate() -> None:
    world = InMemoryOccupancy([])
    common = {
        "world": world,
        "start": (1, 1, 1),
        "goal": (4, 4, 4),
        "extent": (4, 4, 4),
        "directions": offsets_for_mode("discrete_6"),
        "action_mode": "discrete_6",
    }
    planner = validate_task_feasibility(
        **common, maximum_search_nodes=1, maximum_steps=100
    )
    episode = validate_task_feasibility(
        **common,
        maximum_search_nodes=1_000,
        maximum_steps=6,
        recovery_margin_steps=1,
    )

    assert planner.rejection_reason == "planner_budget_exhausted"
    assert episode.rejection_reason == "episode_budget_exceeded"


def test_multi_agent_validation_fails_explicitly() -> None:
    with pytest.raises(NotImplementedError, match="single-agent"):
        MultiVoxelEnv(
            {
                "geometry_validation": {
                    "enabled": True,
                    "maximum_search_nodes": 100,
                }
            }
        )


def test_straight_vertical_and_detour_descriptors_come_from_planned_path() -> None:
    straight = validate_task_feasibility(
        InMemoryOccupancy([]),
        start=(1, 2, 2),
        goal=(4, 2, 2),
        extent=(5, 5, 5),
        directions=offsets_for_mode("discrete_6"),
        action_mode="discrete_6",
        maximum_search_nodes=1_000,
        maximum_steps=20,
    )
    vertical = validate_task_feasibility(
        InMemoryOccupancy([]),
        start=(2, 2, 1),
        goal=(2, 2, 4),
        extent=(5, 5, 5),
        directions=offsets_for_mode("discrete_6"),
        action_mode="discrete_6",
        maximum_search_nodes=1_000,
        maximum_steps=20,
    )
    detour = validate_task_feasibility(
        InMemoryOccupancy([(2, 2, 2)]),
        start=(1, 2, 2),
        goal=(3, 2, 2),
        extent=(5, 5, 5),
        directions=offsets_for_mode("discrete_6"),
        action_mode="discrete_6",
        maximum_search_nodes=1_000,
        maximum_steps=20,
    )

    assert straight.difficulty.detour_ratio == 1.0
    assert straight.difficulty.direction_changes == 0
    assert vertical.difficulty.vertical_displacement == 3
    assert detour.difficulty.shortest_path_length == 4
    assert detour.difficulty.detour_ratio == 2.0
    assert detour.difficulty.direction_changes >= 2


def test_difficulty_bands_filter_without_replanning() -> None:
    band = RoutingDifficultyBand(
        name="detour",
        detour_ratio={"minimum": 1.5},
    )
    result = validate_task_feasibility(
        InMemoryOccupancy([]),
        start=(1, 1, 1),
        goal=(2, 1, 1),
        extent=(3, 3, 3),
        directions=offsets_for_mode("discrete_6"),
        action_mode="discrete_6",
        maximum_search_nodes=100,
        maximum_steps=10,
        difficulty_bands=(band,),
        accepted_difficulty_bands=("detour",),
    )

    assert result.rejection_reason == "difficulty_band_rejected"
    assert result.difficulty_band is None
    assert result.path_length == 1


@pytest.mark.parametrize(
    "band",
    [
        {"name": "empty"},
        {"name": "contradictory", "path_length": {"minimum": 5, "maximum": 4}},
    ],
)
def test_empty_and_contradictory_difficulty_bands_are_rejected(band) -> None:
    with pytest.raises(ValidationError):
        GeometryValidationConfig(difficulty_bands=[band])


def test_routing_descriptors_are_preserved_in_trajectory_artifact() -> None:
    class _Env:
        def close(self):
            return None

    descriptors = {"detour_ratio": 2.0, "shortest_path_length": 4}
    state = _VoxelEpisodeState(
        env_config={"grid_size": 5},
        env=_Env(),
        obs={},
        init_filled=[],
        start_pos=(1, 1, 1),
        goal_pos=(3, 1, 1),
        prev_voxel_count=0,
        steps=[],
        world=None,
        overlay={},
        initial_info={
            "geometry_feasibility": {
                "routing_difficulty": descriptors,
                "difficulty_band": "detour",
            }
        },
    )

    episode = state.finish()

    assert episode.routing_difficulty == descriptors
    assert episode.difficulty_band == "detour"
