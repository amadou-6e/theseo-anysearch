"""Action-space construction and canonical voxel-movement encoding."""
from __future__ import annotations
from itertools import product
from math import sqrt
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
    """Return canonical movement vectors selectable by a discrete mode."""
    return _OFFSETS_BY_MODE.get(mode, ACTION_OFFSETS_26)


def maximum_movement_distance(mode: str) -> float:
    """Return the largest Euclidean displacement selectable by ``mode``."""
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