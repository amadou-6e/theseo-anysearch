"""Policy perception settings and legacy observation migration."""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from theseo_anysearch.environments.perception import get_perception


LegacyObservationMode = Literal["scalar", "box", "radial", "hierarchical_box"]


class _PreflightVoxelEnvironment:
    @staticmethod
    def _has_goal() -> bool:
        return True


class PerceptionSelector(BaseModel):
    """One named sensor and its implementation-specific parameters."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, description="Registered perception name.")
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_registration(self) -> "PerceptionSelector":
        implementation = get_perception(self.name)
        implementation(self.parameters).observation_space(_PreflightVoxelEnvironment())
        self.name = self.name.lower()
        return self


class ObservationConfig(BaseModel):
    """Ordered perception composition with compatibility for legacy modes."""

    model_config = ConfigDict(extra="forbid")
    sensors: list[PerceptionSelector] | None = Field(
        None, description="Ordered named perceptions composing the policy observation."
    )
    mode: LegacyObservationMode | None = Field(None, description="Deprecated; use sensors.")
    box_radius: int = Field(default=2, ge=0, description="Deprecated box-mode radius.")
    box_radii: list[int] | None = Field(None, description="Deprecated hierarchical-box radii.")
    ray_max_len: int = Field(default=16, ge=1, description="Deprecated radial maximum length.")

    @model_validator(mode="before")
    @classmethod
    def warn_legacy_mode(cls, value: Any) -> Any:
        if isinstance(value, dict) and "mode" in value:
            warnings.warn(
                "observation.mode/obs_mode is deprecated; configure observation.sensors instead",
                DeprecationWarning,
                stacklevel=3,
            )
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> "ObservationConfig":
        if self.sensors is not None and self.mode is not None:
            raise ValueError("observation.sensors cannot be combined with deprecated observation.mode")
        selectors = self.resolved_selectors()
        names = [selector.name for selector in selectors]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate perceptions are not allowed: {duplicates}")
        feature_names: set[str] = set()
        preflight = _PreflightVoxelEnvironment()
        for selector in selectors:
            sensor = get_perception(selector.name)(selector.parameters)
            for feature in sensor.observation_space(preflight):
                if feature in feature_names:
                    raise ValueError(f"perceptions produce duplicate feature {feature!r}")
                feature_names.add(feature)
        return self

    def resolved_selectors(self) -> list[PerceptionSelector]:
        if self.sensors is not None:
            return list(self.sensors)
        mode = self.mode or "scalar"
        selectors: list[PerceptionSelector] = []
        if mode == "box":
            selectors.append(PerceptionSelector(name="local_box", parameters={"radius": self.box_radius}))
        elif mode == "radial":
            selectors.append(PerceptionSelector(name="radial", parameters={"max_len": self.ray_max_len}))
        elif mode == "hierarchical_box":
            selectors.append(PerceptionSelector(name="hierarchical_box", parameters={"radii": self.box_radii or [1, 4]}))
        selectors.append(PerceptionSelector(name="goal_vector"))
        return selectors

    def runtime_selectors(self) -> list[dict[str, Any]]:
        return [selector.model_dump(mode="json") for selector in self.resolved_selectors()]

    def metadata(self) -> list[dict[str, Any]]:
        return [{**get_perception(selector.name).metadata.as_dict(), "parameters": dict(selector.parameters)} for selector in self.resolved_selectors()]
