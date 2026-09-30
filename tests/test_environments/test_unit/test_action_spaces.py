import numpy as np
import pytest
from gymnasium import spaces

from theseo_anysearch.environments.action_spaces import (
    ACTION_OFFSETS_26,
    NOOP_ACTION_INDEX,
    build_action_space,
    encode_action,
    maximum_movement_distance,
    offsets_for_mode,
    shortest_action_indices,
    shortest_actions,
)
from theseo_anysearch.models import ActionConfig


@pytest.mark.parametrize("mode,size", [("discrete_6", 6), ("discrete_18", 18), ("discrete_26", 26)])
def test_discrete_action_space_cardinality(mode: str, size: int) -> None:
    action_space = build_action_space(mode)
    assert isinstance(action_space, spaces.Discrete)
    assert action_space.n == size
    assert len(offsets_for_mode(mode)) == size


def test_reduced_offsets_obey_squared_length_limit() -> None:
    assert all(sum(v * v for v in offset) <= 1 for offset in offsets_for_mode("discrete_6"))
    assert all(sum(v * v for v in offset) <= 2 for offset in offsets_for_mode("discrete_18"))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("discrete_6", 1.0),
        ("discrete_18", np.sqrt(2.0)),
        ("discrete_26", np.sqrt(3.0)),
        ("vector_3", np.sqrt(3.0)),
    ],
)
def test_maximum_movement_distance_matches_action_space(mode: str, expected: float) -> None:
    assert maximum_movement_distance(mode) == pytest.approx(expected)


def test_reduced_action_encodes_to_canonical_direction() -> None:
    for mode in ("discrete_6", "discrete_18"):
        for index, offset in enumerate(offsets_for_mode(mode)):
            assert ACTION_OFFSETS_26[encode_action(index, mode)] == offset


def test_vector_3_space_and_encoding() -> None:
    action_space = build_action_space("vector_3")
    assert isinstance(action_space, spaces.MultiDiscrete)
    assert np.array_equal(action_space.nvec, [3, 3, 3])
    assert ACTION_OFFSETS_26[encode_action([2, 1, 0], "vector_3")] == (1, 0, -1)
    assert encode_action([1, 1, 1], "vector_3") == NOOP_ACTION_INDEX


@pytest.mark.parametrize("mode", ["discrete_6", "discrete_18", "vector_3"])
def test_action_modes_validate_from_yaml_schema(mode: str) -> None:
    assert ActionConfig(mode=mode).mode == mode


def test_offsets_for_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="discrete_27"):
        offsets_for_mode("discrete_27")


@pytest.mark.parametrize("mode,size", [("discrete_6", 6), ("discrete_18", 18), ("discrete_26", 26)])
def test_offsets_for_known_mode_resolves(mode: str, size: int) -> None:
    assert len(offsets_for_mode(mode)) == size


@pytest.mark.parametrize("mode", ["discrete_6", "discrete_18", "discrete_26"])
def test_shortest_action_indices_reach_goal_at_exact_distance(mode: str) -> None:
    start = (3, 8, 4)
    goal = (9, 3, 11)

    actions = shortest_action_indices(start, goal, mode)
    offsets = offsets_for_mode(mode)
    cursor = list(start)
    for action in actions:
        for axis, delta in enumerate(offsets[action]):
            cursor[axis] += delta

    from theseo_anysearch.environments.action_spaces import action_step_distance

    assert tuple(cursor) == goal
    assert len(actions) == action_step_distance(start, goal, mode)


def test_shortest_vector_actions_use_multidiscrete_encoding() -> None:
    actions = shortest_actions((0, 0, 0), (2, -1, 1), "vector_3")

    assert actions == ((2, 0, 2), (2, 1, 1))
