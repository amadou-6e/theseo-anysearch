"""Content-addressed task identity and strict suite reuse tests."""

from pathlib import Path

import pytest

from theseo_anysearch.environments.task_identity import (
    accepted_task_manifest,
    build_evaluation_suite,
    configured_geometry_identity,
    geometry_content_identity,
    publish_or_load_evaluation_suite,
)
from theseo_anysearch.environments.validation import (
    GeometryValidationResult,
    TaskFeasibilityResult,
)


def _manifest(*, seed=1, action_mode="discrete_26", transformation=None, band="easy"):
    return accepted_task_manifest(
        coordinates=[(2, 2, 2), (3, 2, 2)],
        seed=seed,
        start=(1, 1, 1),
        route=[(4, 4, 4)],
        action_mode=action_mode,
        transformations=transformation or {"paste_boxes": {"num_boxes": 2}},
        planner_settings={"maximum_search_nodes": 100},
        geometry_validation=GeometryValidationResult(valid=True, coordinate_count=2),
        task_feasibility=TaskFeasibilityResult(
            feasible=True,
            path=((1, 1, 1), (2, 2, 2), (4, 4, 4)),
            path_length=2,
            difficulty_band=band,
        ),
    )


def test_geometry_identity_ignores_source_path_and_iteration_order(tmp_path: Path) -> None:
    coordinates = [(3, 2, 1), (1, 2, 3)]

    assert geometry_content_identity(coordinates) == geometry_content_identity(
        reversed(coordinates)
    )
    first = tmp_path.joinpath("first.stl")
    second = tmp_path.joinpath("nested", "second.stl")
    second.parent.mkdir()
    first.write_bytes(b"same geometry")
    second.write_bytes(b"same geometry")
    assert configured_geometry_identity({"stl_path": first}) == (
        configured_geometry_identity({"stl_path": second})
    )


def test_composed_source_content_changes_geometry_identity(tmp_path: Path) -> None:
    first = tmp_path.joinpath("first.stl")
    second = tmp_path.joinpath("second.stl")
    first.write_bytes(b"first geometry")
    second.write_bytes(b"second geometry")
    base = {"geometry_sources": [{"type": "stl", "path": str(first)}]}
    changed_stl = {"geometry_sources": [{"type": "stl", "path": str(second)}]}
    changed_box = {
        "geometry_sources": [
            {"type": "boxes", "boxes": [[1, 1, 1, 2, 2, 2]]}
        ]
    }

    assert configured_geometry_identity(base) != configured_geometry_identity(changed_stl)
    assert configured_geometry_identity(base) != configured_geometry_identity(changed_box)


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 2},
        {"action_mode": "discrete_6"},
        {"transformation": {"paste_boxes": {"num_boxes": 3}}},
    ],
)
def test_semantic_changes_invalidate_task_identity(change) -> None:
    assert _manifest().identity_sha256 != _manifest(**change).identity_sha256


@pytest.mark.parametrize(
    "planner_settings",
    [
        {"maximum_search_nodes": 100, "clearance_radius": 3},
        {
            "maximum_search_nodes": 100,
            "accepted_difficulty_bands": ["easy", "hard"],
        },
    ],
)
def test_validator_settings_invalidate_task_identity(planner_settings) -> None:
    baseline = _manifest()
    changed = accepted_task_manifest(
        coordinates=[(2, 2, 2), (3, 2, 2)],
        seed=1,
        start=(1, 1, 1),
        route=[(4, 4, 4)],
        action_mode="discrete_26",
        transformations={"paste_boxes": {"num_boxes": 2}},
        planner_settings=planner_settings,
        geometry_validation=baseline.geometry_validation,
        task_feasibility=baseline.task_feasibility,
    )

    assert baseline.identity_sha256 != changed.identity_sha256


def test_evaluation_suite_reuses_exact_membership_and_rejects_changes(tmp_path: Path) -> None:
    path = tmp_path.joinpath("evaluation_suite.json")
    expected = build_evaluation_suite([_manifest(seed=1), _manifest(seed=2, band="hard")])

    first = publish_or_load_evaluation_suite(path, expected)
    second = publish_or_load_evaluation_suite(path, expected)

    assert first.identity_sha256 == second.identity_sha256
    assert second.difficulty_distribution == {"easy": 1, "hard": 1}
    with pytest.raises(ValueError, match="suite identity mismatch"):
        publish_or_load_evaluation_suite(
            path, build_evaluation_suite([_manifest(seed=3)])
        )
