"""Strict scenario models and validation for policy explanations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field, model_validator


Coordinate = tuple[int, int, int]


class ScenarioModel(BaseModel):
    """Base model that rejects misspelled or unsupported scenario fields."""

    model_config = ConfigDict(extra="forbid")


class ScenarioExecution(ScenarioModel):
    """How an environment scenario is executed."""

    mode: Literal["single_step", "rollout", "actions"] = "single_step"
    max_steps: int = Field(default=50, ge=1)
    actions: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_actions(self) -> "ScenarioExecution":
        """Require actions only for explicit action replay."""

        if self.mode == "actions" and not self.actions:
            raise ValueError("execution.mode 'actions' requires execution.actions")
        if self.mode != "actions" and self.actions:
            raise ValueError("execution.actions is only valid with execution.mode 'actions'")
        return self


class EnvironmentState(ScenarioModel):
    """Concrete voxel state layered over the run's environment settings."""

    cursor: Coordinate
    route: tuple[Coordinate, ...] = Field(min_length=1)
    geometry_boxes: tuple[tuple[int, int, int, int, int, int], ...] = ()
    trail: tuple[Coordinate, ...] = ()


class EnvironmentScenario(ScenarioModel):
    """Environment-backed and therefore validity-checked scenario."""

    type: Literal["environment"]
    seed: int = 142
    state: EnvironmentState
    execution: ScenarioExecution = Field(default_factory=ScenarioExecution)


class ObservationScenario(ScenarioModel):
    """Possibly fictional policy observation."""

    type: Literal["observation"]
    observation: dict[str, list[float]]
    chosen_action: Literal["policy"] | int = "policy"


Scenario = Annotated[EnvironmentScenario | ObservationScenario, Field(discriminator="type")]


class ScenarioDocument(ScenarioModel):
    """Typed wrapper used to validate a discriminated scenario document."""

    scenario: Scenario


def load_scenario(path: Path) -> EnvironmentScenario | ObservationScenario:
    """Load one strict YAML scenario."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"scenario {path} must contain a YAML mapping")
    return ScenarioDocument.model_validate({"scenario": raw}).scenario


def validate_observation(
    values: dict[str, list[float]],
    observation_space: spaces.Space,
) -> dict[str, np.ndarray]:
    """Validate exact keys, shapes, finiteness, and bounds against a policy space."""

    if not isinstance(observation_space, spaces.Dict):
        raise ValueError("fictional observations require a dictionary observation space")
    expected = set(observation_space.spaces)
    supplied = set(values)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        raise ValueError(
            f"observation schema mismatch: missing={missing}, unknown={unknown}"
        )
    result: dict[str, np.ndarray] = {}
    for name, field_space in observation_space.spaces.items():
        array = np.asarray(values[name], dtype=np.float32)
        if array.shape != field_space.shape:
            raise ValueError(
                f"observation field {name!r} has shape {array.shape}, "
                f"expected {field_space.shape}"
            )
        if not all(math.isfinite(float(value)) for value in array.reshape(-1)):
            raise ValueError(f"observation field {name!r} contains a non-finite value")
        if not field_space.contains(array):
            raise ValueError(f"observation field {name!r} is outside its declared bounds")
        result[name] = array
    return result
