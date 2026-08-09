"""Named perception registry and composition pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from gymnasium import spaces

from theseo_anysearch.environments.perception.base import Perception


_PERCEPTIONS: dict[str, type[Perception]] = {}


def register_perception(name: str, implementation: type[Perception]) -> None:
    """Register a perception implementation without changing trainer code."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("perception name cannot be empty")
    if normalized in _PERCEPTIONS:
        raise ValueError(f"perception {normalized!r} is already registered")
    if implementation.metadata.name != normalized:
        raise ValueError(
            f"perception metadata name {implementation.metadata.name!r} "
            f"does not match registration name {normalized!r}"
        )
    _PERCEPTIONS[normalized] = implementation


def perception_names() -> tuple[str, ...]:
    return tuple(sorted(_PERCEPTIONS))


def get_perception(name: str) -> type[Perception]:
    try:
        return _PERCEPTIONS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown perception {name!r}; registered perceptions: {list(perception_names())}"
        ) from None


class PerceptionPipeline:
    """Ordered collection of sensors producing one structured observation."""

    def __init__(self, selectors: Iterable[Mapping[str, Any]]) -> None:
        self.sensors = []
        names: set[str] = set()
        for selector in selectors:
            name = str(selector["name"]).lower()
            if name in names:
                raise ValueError(f"duplicate perception {name!r}")
            names.add(name)
            self.sensors.append(get_perception(name)(selector.get("parameters")))

    def observation_space(self, env: Any) -> spaces.Dict:
        features: dict[str, Any] = {}
        for sensor in self.sensors:
            for key, space in sensor.observation_space(env).items():
                if key in features:
                    raise ValueError(
                        f"perceptions produce duplicate observation feature {key!r}"
                    )
                features[key] = space
        return spaces.Dict(features)

    def compute(self, env: Any, native_observation: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for sensor in self.sensors:
            values = sensor.compute(env, native_observation)
            overlap = result.keys() & values.keys()
            if overlap:
                raise ValueError(
                    f"perceptions produced duplicate observation features: {sorted(overlap)}"
                )
            result.update(values)
        return result

    def metadata(self) -> list[dict[str, Any]]:
        return [
            {**sensor.metadata.as_dict(), "parameters": dict(sensor.parameters)}
            for sensor in self.sensors
        ]

