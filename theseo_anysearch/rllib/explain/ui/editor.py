"""Schema-driven editing of policy observations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import yaml
from gymnasium import spaces

from theseo_anysearch.rllib.explain.scenarios import validate_observation


Axis = Literal["x", "y", "z"]


class ObservationEditor:
    """Mutable observation document constrained by a Gymnasium Dict space."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        observation: Mapping[str, object],
    ) -> None:
        self.observation_space = observation_space
        self._values = validate_observation(
            {name: np.asarray(value).tolist() for name, value in observation.items()},
            observation_space,
        )

    @property
    def values(self) -> dict[str, np.ndarray]:
        """Return a detached copy of the current validated observation."""

        return {name: value.copy() for name, value in self._values.items()}

    def field_bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds for one observation field."""

        field = self.observation_space.spaces[name]
        return np.asarray(field.low), np.asarray(field.high)

    def set_field(self, name: str, value: object) -> None:
        """Replace one field and validate the complete observation."""

        candidate = self.values
        candidate[name] = np.asarray(value, dtype=np.float32)
        self._values = validate_observation(
            {key: item.tolist() for key, item in candidate.items()},
            self.observation_space,
        )

    @property
    def box_side(self) -> int:
        """Return the side length of the cubic local grid."""

        size = int(self._values["local_grid"].size)
        side = round(size ** (1.0 / 3.0))
        if side**3 != size:
            raise ValueError(f"local_grid size {size} is not a cube")
        return side

    def slice(self, axis: Axis, index: int) -> np.ndarray:
        """Return a detached two-dimensional local-grid slice."""

        cube = self._cube()
        self._validate_slice_index(index)
        if axis == "x":
            return cube[index, :, :].copy()
        if axis == "y":
            return cube[:, index, :].copy()
        if axis == "z":
            return cube[:, :, index].copy()
        raise ValueError(f"unsupported slice axis: {axis!r}")

    def set_slice(self, axis: Axis, index: int, values: object) -> None:
        """Replace one local-grid slice using the Rust x/y/z flattening order."""

        self._validate_slice_index(index)
        array = np.asarray(values, dtype=np.float32)
        expected = (self.box_side, self.box_side)
        if array.shape != expected:
            raise ValueError(f"slice has shape {array.shape}, expected {expected}")
        cube = self._cube().copy()
        if axis == "x":
            cube[index, :, :] = array
        elif axis == "y":
            cube[:, index, :] = array
        elif axis == "z":
            cube[:, :, index] = array
        else:
            raise ValueError(f"unsupported slice axis: {axis!r}")
        self.set_field("local_grid", cube.reshape(-1))

    def save_scenario(self, path: Path, chosen_action: str | int = "policy") -> None:
        """Write a strict fictional-observation scenario YAML."""

        payload = self.to_scenario(chosen_action)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def to_scenario(self, chosen_action: str | int = "policy") -> dict:
        """Return the strict fictional-observation scenario mapping."""

        return {
            "type": "observation",
            "chosen_action": chosen_action,
            "observation": {
                name: value.tolist() for name, value in self._values.items()
            },
        }

    def _cube(self) -> np.ndarray:
        return self._values["local_grid"].reshape(
            self.box_side, self.box_side, self.box_side
        )

    def _validate_slice_index(self, index: int) -> None:
        if index < 0 or index >= self.box_side:
            raise ValueError(
                f"slice index {index} is outside [0, {self.box_side - 1}]"
            )
