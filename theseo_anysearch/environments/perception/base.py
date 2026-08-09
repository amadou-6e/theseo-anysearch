"""Stable contracts for composable policy perception."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import gymnasium
from pydantic import BaseModel, ConfigDict


class PerceptionParameters(BaseModel):
    """Default no-parameter sensor configuration."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class PerceptionMetadata:
    """Machine-readable compatibility and resource metadata for a sensor."""

    name: str
    schema_version: str = "1"
    dtype: str = "float32"
    normalization: str = "none"
    required_capabilities: tuple[str, ...] = ()
    compatible_models: tuple[str, ...] = ()
    estimated_compute: str = "constant"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Perception(ABC):
    """A deterministic named sensor with a declared observation space."""

    metadata: PerceptionMetadata
    parameter_model: type[PerceptionParameters] = PerceptionParameters

    def __init__(self, parameters: Mapping[str, Any] | None = None) -> None:
        self.config = self.parameter_model.model_validate(parameters or {})

    @property
    def parameters(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")

    @abstractmethod
    def observation_space(self, env: Any) -> Mapping[str, gymnasium.Space]:
        """Declare the structured features emitted for an environment."""

    @abstractmethod
    def compute(self, env: Any, native_observation: Any) -> Mapping[str, Any]:
        """Compute features without mutating environment state."""
