"""Action-space construction and canonical voxel-movement encoding."""
from __future__ import annotations
from itertools import product
from math import ceil, sqrt
from typing import Any
import numpy as np
from gymnasium import spaces

ACTION_OFFSETS_26 = tuple(offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0))
ACTION_OFFSETS_6 = tuple(offset for offset in ACTION_OFFSETS_26 if sum(value * value for value in offset) <= 1)
ACTION_OFFSETS_18 = tuple(offset for offset in ACTION_OFFSETS_26 if sum(value * value for value in offset) <= 2)
NOOP_ACTION_INDEX = 26
_OFFSET_TO_ACTION = {offset: index for index, offset in enumerate(ACTION_OFFSETS_26)}
_OFFSETS_BY_MODE = {
    "discrete_6": ACTION_OFFSETS_6,
    "discrete_18": ACTION_OFFSETS_18,
    "discrete_26": ACTION_OFFSETS_26,
}

def offsets_for_mode(mode: str) -> tuple[tuple[int, int, int], ...]:
    """Return canonical movement vectors selectable by a discrete mode.

    Raises
    ------
    ValueError
        If ``mode`` is not a registered discrete action mode.
    """
    try:
        return _OFFSETS_BY_MODE[mode]
    except KeyError:
        valid = sorted(_OFFSETS_BY_MODE)
        raise ValueError(
            f"Unknown action mode: {mode!r}. Valid discrete modes: {valid}"
        ) from None


def action_step_distance(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    mode: str,
) -> int:
    """Return the exact empty-grid shortest path in configured action steps."""
    deltas = [abs(goal[index] - start[index]) for index in range(3)]
    if mode == "discrete_6":
        return sum(deltas)
    if mode == "discrete_18":
        return max(max(deltas), ceil(sum(deltas) / 2))
    if mode in {"discrete_26", "vector_3"}:
        return max(deltas)
    raise ValueError(f"unsupported action mode for waypoint route: {mode!r}")


def shortest_action_indices(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    mode: str,
) -> tuple[int, ...]:
    """Return a deterministic shortest empty-grid action sequence."""

    offsets = offsets_for_mode(mode)
    index_by_offset = {offset: index for index, offset in enumerate(offsets)}
    remaining = [goal[index] - start[index] for index in range(3)]
    axes_per_step = {"discrete_6": 1, "discrete_18": 2, "discrete_26": 3}[mode]
    actions: list[int] = []
    while any(remaining):
        active_axes = sorted(
            (index for index, value in enumerate(remaining) if value),
            key=lambda index: (-abs(remaining[index]), index),
        )[:axes_per_step]
        offset = [0, 0, 0]
        for axis in active_axes:
            offset[axis] = 1 if remaining[axis] > 0 else -1
            remaining[axis] -= offset[axis]
        actions.append(index_by_offset[tuple(offset)])
    return tuple(actions)

def maximum_movement_distance(mode: str) -> float:
    """Return the largest Euclidean displacement selectable by ``mode``."""
    if mode == "vector_3":
        return sqrt(3.0)
    offsets = offsets_for_mode(mode)
    return max(sqrt(sum(value * value for value in offset)) for offset in offsets)


def build_action_space(mode: str) -> spaces.Space:
    """Build the Gymnasium action space configured by ``mode``."""
    if mode == "vector_3":
        return spaces.MultiDiscrete([3, 3, 3])
    if mode in {"discrete_6", "discrete_18", "discrete_26", "vector_3"}:
        return spaces.Discrete(len(offsets_for_mode(mode)))
    raise ValueError(f"Unknown action mode: {mode!r}")

def encode_action(action: Any, mode: str) -> int:
    """Encode a configured action as a canonical Rust movement index."""
    if mode == "vector_3":
        values = np.asarray(action, dtype=np.int64)
        if values.shape != (3,) or np.any(values < 0) or np.any(values > 2):
            return NOOP_ACTION_INDEX + 1
        offset = tuple(int(value) - 1 for value in values)
        return NOOP_ACTION_INDEX if offset == (0, 0, 0) else _OFFSET_TO_ACTION[offset]
    try:
        index = int(action)
    except (TypeError, ValueError):
        return NOOP_ACTION_INDEX + 1
    offsets = offsets_for_mode(mode)
    if index < 0 or index >= len(offsets):
        return NOOP_ACTION_INDEX + 1
    return _OFFSET_TO_ACTION[offsets[index]]
