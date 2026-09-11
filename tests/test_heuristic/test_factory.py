"""Factory compatibility tests for YAML-selectable voxel heuristics."""

import pytest

from theseo_anysearch.heuristic import (
    VoxelAStarOracle,
    VoxelDijkstraHeuristic,
    VoxelReplanningAStarHeuristic,
    VoxelWeightedAStarHeuristic,
    build_voxel_heuristic,
)
from tests.test_environments.test_integration._voxel_validity_support import (
    make_radial_test_env,
)


@pytest.mark.parametrize(
    ("yaml_name", "expected_type"),
    [
        ("astar", VoxelAStarOracle),
        ("dijkstra", VoxelDijkstraHeuristic),
        ("weighted_astar", VoxelWeightedAStarHeuristic),
        ("replanning_astar", VoxelReplanningAStarHeuristic),
    ],
)
def test_factory_preserves_yaml_names(tmp_path, yaml_name, expected_type):
    """Each documented YAML name constructs its corresponding strategy."""
    env = make_radial_test_env(tmp_path)

    heuristic = build_voxel_heuristic(env, yaml_name, weight=2.0)

    assert isinstance(heuristic, expected_type)


def test_factory_rejects_unknown_yaml_name(tmp_path):
    """Unknown YAML names fail with a configuration-oriented error."""
    env = make_radial_test_env(tmp_path)

    with pytest.raises(ValueError, match="Unknown voxel heuristic type"):
        build_voxel_heuristic(env, "unknown")
