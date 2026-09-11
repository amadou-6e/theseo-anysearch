"""Observation feature schemas for explainability backends."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict


OBSERVATION_ORDER = (
    "cursor_pos",
    "goal_direction",
    "goal_distance",
    "local_grid",
    "ray_hits",
    "ray_hit_types",
    "steps_remaining",
)


def action_directions_26() -> tuple[tuple[int, int, int], ...]:
    """Return directions aligned with the voxel 26-action space."""

    directions: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                directions.append((dx, dy, dz))
    return tuple(directions)


class FeatureGroupSchema(BaseModel):
    """Flat feature metadata for one observation group.

    Parameters
    ----------
    name : str
        Observation group name.
    shape : tuple[int, ...]
        Original array shape.
    start : int
        Inclusive flat start index.
    stop : int
        Exclusive flat stop index.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def slice(self) -> slice:
        """Return the flat slice occupied by this group."""

        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        """Return the number of flat features in this group."""

        return self.stop - self.start


class FeatureSchema:
    """Stable flattening schema for dictionary observations.

    Parameters
    ----------
    groups : Sequence[FeatureGroupSchema]
        Feature groups in flat-vector order.
    action_directions : Sequence[tuple[int, int, int]], optional
        Action direction metadata for action-aligned feature names.
    """

    version = 1

    def __init__(
        self,
        groups: Sequence[FeatureGroupSchema],
        action_directions: Sequence[tuple[int, int, int]] = action_directions_26(),
    ) -> None:
        self._groups = tuple(groups)
        self._groups_by_name = {group.name: group for group in self._groups}
        self._action_directions = tuple(action_directions)

    @classmethod
    def from_observation(
        cls,
        observation: Mapping[str, object],
        *,
        action_directions: Sequence[tuple[int, int, int]] | None = None,
    ) -> "FeatureSchema":
        """Build a schema from one observation dictionary."""

        start = 0
        groups: list[FeatureGroupSchema] = []
        names = [name for name in OBSERVATION_ORDER if name in observation]
        unexpected = sorted(set(observation) - set(names))
        if unexpected:
            raise ValueError(f"observation contains unsupported groups: {unexpected}")
        if not names:
            raise ValueError("observation contains no supported feature groups")
        for name in names:
            array = np.asarray(observation[name], dtype=np.float32)
            stop = start + int(array.size)
            groups.append(FeatureGroupSchema(name=name, shape=tuple(array.shape), start=start, stop=stop))
            start = stop
        return cls(
            groups,
            action_directions=(
                action_directions
                if action_directions is not None
                else action_directions_26()
            ),
        )

    @classmethod
    def from_observation_space(cls, space: object) -> "FeatureSchema":
        """Build a schema from a Gymnasium-like dictionary observation space."""

        spaces = getattr(space, "spaces", None)
        if spaces is None:
            raise ValueError("observation space must expose a spaces mapping")
        sample = {
            name: np.zeros(getattr(spaces[name], "shape"), dtype=np.float32)
            for name in OBSERVATION_ORDER
            if name in spaces
        }
        return cls.from_observation(sample)

    @property
    def groups(self) -> tuple[FeatureGroupSchema, ...]:
        """Return feature groups in flat-vector order."""

        return self._groups

    @property
    def action_directions(self) -> tuple[tuple[int, int, int], ...]:
        """Return action directions aligned with ray features."""

        return self._action_directions

    @property
    def size(self) -> int:
        """Return total flat feature count."""

        return self._groups[-1].stop if self._groups else 0

    def flatten(self, observation: Mapping[str, object]) -> np.ndarray:
        """Flatten one observation dictionary."""

        chunks: list[np.ndarray] = []
        for group in self._groups:
            array = np.asarray(observation[group.name], dtype=np.float32)
            if tuple(array.shape) != group.shape:
                raise ValueError(
                    f"group {group.name!r} has shape {tuple(array.shape)}, expected {group.shape}"
                )
            chunks.append(array.reshape(-1))
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def flatten_batch(self, observations: Sequence[Mapping[str, object]]) -> np.ndarray:
        """Flatten a batch of observation dictionaries."""

        return np.stack([self.flatten(observation) for observation in observations]).astype(
            np.float32,
            copy=False,
        )

    def unflatten(self, row: np.ndarray) -> dict[str, np.ndarray]:
        """Restore a flat feature row to an observation dictionary."""

        flat = np.asarray(row, dtype=np.float32).reshape(-1)
        if flat.size != self.size:
            raise ValueError(f"flat row has {flat.size} features, expected {self.size}")
        return {
            group.name: flat[group.slice].reshape(group.shape).astype(np.float32, copy=False)
            for group in self._groups
        }

    def feature_names(self) -> list[str]:
        """Return stable human-readable flat feature names."""

        names: list[str] = []
        for group in self._groups:
            if group.name in {"ray_hits", "ray_hit_types"} and group.size == len(self._action_directions):
                for action, direction in enumerate(self._action_directions):
                    dx, dy, dz = direction
                    names.append(f"{group.name}[{action}](dx={dx},dy={dy},dz={dz})")
            else:
                for index in range(group.size):
                    names.append(f"{group.name}[{index}]")
        return names

    def group_slices(self) -> dict[str, slice]:
        """Return flat slices keyed by observation group name."""

        return {group.name: group.slice for group in self._groups}

    def action_feature_index(self, group_name: str, action: int) -> int:
        """Return the flat index for an action-aligned feature."""

        group = self._groups_by_name[group_name]
        if action < 0 or action >= group.size:
            raise ValueError(f"action {action} is outside group {group_name!r}")
        return group.start + action

    def local_grid_index(self, direction: tuple[int, int, int]) -> int:
        """Return the flat local-grid index for a relative voxel direction."""
        group = self._groups_by_name["local_grid"]
        side = round(group.size ** (1.0 / 3.0))
        if side**3 != group.size or side % 2 != 1:
            raise ValueError(f"local_grid size {group.size} is not an odd cube")
        radius = side // 2
        dx, dy, dz = direction
        if any(abs(value) > radius for value in direction):
            raise ValueError(
                f"direction {direction} is outside local-grid radius {radius}"
            )
        return ((dx + radius) * side + (dy + radius)) * side + (dz + radius)

    def group_names(self) -> list[str]:
        """Return group names in flat-vector order."""

        return [group.name for group in self._groups]
