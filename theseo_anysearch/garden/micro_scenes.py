"""Hand-checkable deterministic volumes used by the P0 target oracle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MicroScene:
    """One named binary obstacle volume and optional unknown-space mask."""

    name: str
    occupancy: np.ndarray
    unknown_mask: np.ndarray


# Literal P0 oracles: occupied, boundary, free components, graph cycle rank, valid cells.
MICRO_SCENE_ORACLES: dict[str, tuple[int, int, int, int, int]] = {
    "empty": (0, 0, 1, 1216, 729),
    "filled": (729, 0, 0, 0, 729),
    "single_center": (1, 1, 1, 1211, 729),
    "single_corner": (1, 1, 1, 1211, 729),
    "plane_x": (81, 81, 2, 992, 729),
    "plane_y": (81, 81, 2, 992, 729),
    "plane_z": (81, 81, 2, 992, 729),
    "wall_hole_x": (80, 80, 1, 992, 729),
    "wall_hole_y": (80, 80, 1, 992, 729),
    "wall_hole_z": (80, 80, 1, 992, 729),
    "corridor_x": (722, 30, 1, 0, 729),
    "corridor_y": (722, 30, 1, 0, 729),
    "corridor_z": (722, 30, 1, 0, 729),
    "branch_xy": (716, 50, 1, 0, 729),
    "branch_xz": (716, 50, 1, 0, 729),
    "branch_yz": (716, 50, 1, 0, 729),
    "loop_xy": (713, 60, 1, 1, 729),
    "loop_xz": (713, 60, 1, 1, 729),
    "loop_yz": (713, 60, 1, 1, 729),
    "disconnected_x": (675, 99, 2, 56, 729),
    "disconnected_y": (675, 99, 2, 56, 729),
    "disconnected_z": (675, 99, 2, 56, 729),
    "doorway_x": (72, 72, 1, 1012, 729),
    "doorway_y": (72, 72, 1, 1012, 729),
    "doorway_z": (72, 72, 1, 1012, 729),
    "room_closed": (386, 294, 1, 540, 729),
    "room_door_x": (385, 293, 1, 540, 729),
    "room_door_y": (385, 293, 1, 540, 729),
    "room_door_z": (385, 293, 1, 540, 729),
    "unknown_face_x": (0, 0, 1, 1072, 648),
    "unknown_face_y": (0, 0, 1, 1072, 648),
    "unknown_face_z": (0, 0, 1, 1072, 648),
}


def _empty(side: int) -> np.ndarray:
    return np.zeros((side, side, side), dtype=bool)


def _filled(side: int) -> np.ndarray:
    return np.ones((side, side, side), dtype=bool)


def _plane(side: int, axis: int, hole_radius: int | None = None) -> np.ndarray:
    occupancy = _empty(side)
    middle = side // 2
    slices = [slice(None)] * 3
    slices[axis] = middle
    occupancy[tuple(slices)] = True
    if hole_radius is not None:
        other_axes = [candidate for candidate in range(3) if candidate != axis]
        hole = [slice(None)] * 3
        hole[axis] = middle
        for other_axis in other_axes:
            hole[other_axis] = slice(middle - hole_radius, middle + hole_radius + 1)
        occupancy[tuple(hole)] = False
    return occupancy


def _corridor(side: int, axis: int) -> np.ndarray:
    occupancy = _filled(side)
    middle = side // 2
    free = [middle, middle, middle]
    free[axis] = slice(1, side - 1)
    occupancy[tuple(free)] = False
    return occupancy


def _branch(side: int, axes: tuple[int, int]) -> np.ndarray:
    occupancy = _filled(side)
    middle = side // 2
    for axis in axes:
        free = [middle, middle, middle]
        free[axis] = slice(1, side - 1)
        occupancy[tuple(free)] = False
    return occupancy


def _loop(side: int, plane_axes: tuple[int, int]) -> np.ndarray:
    occupancy = _filled(side)
    middle = side // 2
    fixed_axis = ({0, 1, 2} - set(plane_axes)).pop()
    low, high = 2, side - 3
    for position in range(low, high + 1):
        for fixed in (low, high):
            first = [middle, middle, middle]
            first[fixed_axis] = middle
            first[plane_axes[0]] = position
            first[plane_axes[1]] = fixed
            occupancy[tuple(first)] = False
            second = list(first)
            second[plane_axes[0]], second[plane_axes[1]] = fixed, position
            occupancy[tuple(second)] = False
    return occupancy


def _disconnected(side: int, axis: int) -> np.ndarray:
    occupancy = _filled(side)
    for center in (2, side - 3):
        region = [slice(1, 4), slice(1, 4), slice(1, 4)]
        region[axis] = slice(center - 1, center + 2)
        occupancy[tuple(region)] = False
    return occupancy


def _room(side: int, door_axis: int | None = None) -> np.ndarray:
    occupancy = _empty(side)
    occupancy[[0, -1], :, :] = True
    occupancy[:, [0, -1], :] = True
    occupancy[:, :, [0, -1]] = True
    if door_axis is not None:
        middle = side // 2
        door = [middle, middle, middle]
        door[door_axis] = 0
        occupancy[tuple(door)] = False
    return occupancy


def make_micro_scenes(side: int = 9) -> tuple[MicroScene, ...]:
    """Return the fixed 32-scene P0 geometry fixture."""

    if side != 9:
        raise ValueError("the frozen P0 micro-scene oracle is defined only for side 9")
    zero_unknown = _empty(side)
    scenes: list[MicroScene] = [
        MicroScene("empty", _empty(side), zero_unknown.copy()),
        MicroScene("filled", _filled(side), zero_unknown.copy()),
    ]
    single_center = _empty(side)
    single_center[(side // 2,) * 3] = True
    single_corner = _empty(side)
    single_corner[1, 1, 1] = True
    scenes.extend(
        (
            MicroScene("single_center", single_center, zero_unknown.copy()),
            MicroScene("single_corner", single_corner, zero_unknown.copy()),
        )
    )
    for axis, label in enumerate("xyz"):
        scenes.append(MicroScene(f"plane_{label}", _plane(side, axis), zero_unknown.copy()))
    for axis, label in enumerate("xyz"):
        scenes.append(
            MicroScene(f"wall_hole_{label}", _plane(side, axis, hole_radius=0), zero_unknown.copy())
        )
    for axis, label in enumerate("xyz"):
        scenes.append(MicroScene(f"corridor_{label}", _corridor(side, axis), zero_unknown.copy()))
    for axes, label in (((0, 1), "xy"), ((0, 2), "xz"), ((1, 2), "yz")):
        scenes.append(MicroScene(f"branch_{label}", _branch(side, axes), zero_unknown.copy()))
    for axes, label in (((0, 1), "xy"), ((0, 2), "xz"), ((1, 2), "yz")):
        scenes.append(MicroScene(f"loop_{label}", _loop(side, axes), zero_unknown.copy()))
    for axis, label in enumerate("xyz"):
        scenes.append(
            MicroScene(f"disconnected_{label}", _disconnected(side, axis), zero_unknown.copy())
        )
    for axis, label in enumerate("xyz"):
        scenes.append(
            MicroScene(f"doorway_{label}", _plane(side, axis, hole_radius=1), zero_unknown.copy())
        )
    scenes.append(MicroScene("room_closed", _room(side), zero_unknown.copy()))
    for axis, label in enumerate("xyz"):
        scenes.append(MicroScene(f"room_door_{label}", _room(side, axis), zero_unknown.copy()))
    for axis, label in enumerate("xyz"):
        unknown = _empty(side)
        face = [slice(None)] * 3
        face[axis] = 0
        unknown[tuple(face)] = True
        scenes.append(MicroScene(f"unknown_face_{label}", _empty(side), unknown))
    if len(scenes) != 32 or len({scene.name for scene in scenes}) != 32:
        raise AssertionError("the P0 micro-scene fixture must contain 32 unique scenes")
    return tuple(scenes)
