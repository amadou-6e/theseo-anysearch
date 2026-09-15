"""Stable entry-point boundary for optional world-provider wheels."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Literal, Protocol

API_VERSION = 1
ENTRY_POINT_GROUP = "theseo_anysearch.world_providers"
PARAMETER_TYPES = ("integer", "number", "text", "boolean")
ParameterType = Literal["integer", "number", "text", "boolean"]
_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class ProviderParameter:
    name: str
    kind: ParameterType
    required: bool = False
    default: int | float | str | bool | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    help: str = ""

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name) or self.name in {"seed", "output"}:
            raise ValueError("invalid or reserved provider parameter name")
        if self.kind not in PARAMETER_TYPES:
            raise ValueError("unsupported provider parameter type")
        if self.required and self.default is not None:
            raise ValueError("a required parameter cannot have a default")
        if self.kind not in {"integer", "number"} and (self.minimum is not None or self.maximum is not None):
            raise ValueError("only numeric parameters can have bounds")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("parameter minimum exceeds maximum")

    def validate(self, value: object) -> int | float | str | bool:
        if self.kind == "integer" and (type(value) is not int):
            raise ValueError(f"{self.name} requires an integer")
        if self.kind == "number" and (type(value) not in (float, int)):
            raise ValueError(f"{self.name} requires a number")
        if self.kind == "text" and type(value) is not str:
            raise ValueError(f"{self.name} requires text")
        if self.kind == "boolean" and type(value) is not bool:
            raise ValueError(f"{self.name} requires a boolean")
        if self.minimum is not None and value < self.minimum:  # type: ignore[operator]
            raise ValueError(f"{self.name} is below its minimum")
        if self.maximum is not None and value > self.maximum:  # type: ignore[operator]
            raise ValueError(f"{self.name} is above its maximum")
        return value  # type: ignore[return-value]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    version: str
    description: str
    native_meters_per_voxel: float
    parameters: tuple[ProviderParameter, ...] = ()
    api_version: int = API_VERSION
    resolution_parameter: str | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError("invalid provider name")
        if self.api_version != API_VERSION:
            raise ValueError(f"provider {self.name} requires unsupported API {self.api_version}")
        if not math.isfinite(self.native_meters_per_voxel) or self.native_meters_per_voxel <= 0:
            raise ValueError("native resolution must be positive")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate provider parameters")
        if self.resolution_parameter is not None:
            parameter = next((p for p in self.parameters if p.name == self.resolution_parameter), None)
            if parameter is None or parameter.kind != "number":
                raise ValueError("resolution parameter must name a numeric provider parameter")

    def output_resolution(self, parameters: dict[str, object]) -> float:
        value = self.native_meters_per_voxel
        if self.resolution_parameter is not None:
            value = float(parameters[self.resolution_parameter])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("output resolution must be positive and finite")
        return value


@dataclass(frozen=True)
class GenerationSummary:
    rejected_task_strata: tuple[str, ...] = ()


class WorldProvider(Protocol):
    @property
    def info(self) -> ProviderInfo: ...

    def generate(self, *, seed: int, output: Path, parameters: dict[str, object]) -> GenerationSummary:
        """Write occupancy.npy and existing routing-manifest sidecars to output."""


def _provider_from_entry_point(point: object) -> WorldProvider:
    loaded = point.load()  # type: ignore[attr-defined]
    provider = loaded() if isinstance(loaded, type) else loaded
    info = provider.info
    if info.name != point.name:  # type: ignore[attr-defined]
        raise ValueError("entry-point name differs from provider info")
    return provider


def provider_errors() -> dict[str, str]:
    """Report broken optional plugins without taking down unrelated commands."""

    errors = {}
    for point in entry_points(group=ENTRY_POINT_GROUP):
        if point.name == "fixture-boxes":
            errors[point.name] = "entry-point name conflicts with the built-in fixture"
            continue
        try:
            _provider_from_entry_point(point)
        except Exception as exc:
            errors[point.name] = str(exc)
    return errors


def installed_providers() -> dict[str, WorldProvider]:
    from theseo_anysearch.world_providers.fixtures import FixtureBoxesProvider

    providers: dict[str, WorldProvider] = {"fixture-boxes": FixtureBoxesProvider()}
    for point in entry_points(group=ENTRY_POINT_GROUP):
        if point.name in providers:
            continue
        try:
            provider = _provider_from_entry_point(point)
        except Exception:
            continue
        providers[provider.info.name] = provider
    return providers


def load_provider(name: str) -> WorldProvider:
    provider = installed_providers().get(name)
    if provider is None:
        error = provider_errors().get(name)
        if error is not None:
            raise ValueError(f"world provider {name!r} is incompatible or broken: {error}")
        raise ValueError(f"world provider {name!r} is not installed; run 'anysearch worlds list'")
    return provider
