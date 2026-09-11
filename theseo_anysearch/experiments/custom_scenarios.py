"""Convention-based episode scenario discovery and execution."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import inspect
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomScenarioError(ValueError):
    """Raised when a scenario provider violates its explicit contract."""


@runtime_checkable
class ScenarioWorld(Protocol):
    """Continuous-grid view; storage regions and cache residency stay hidden."""

    extent: tuple[int, int, int]
    identity: str | None

    def occupied(self, coordinate: tuple[int, int, int]) -> bool: ...

    def occupied_in_region(
        self,
        minimum: tuple[int, int, int],
        maximum_exclusive: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], ...]: ...


class ScenarioContext(BaseModel):
    """Immutable reset inputs supplied to an episode scenario provider."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    seed: int
    episode_index: int
    scope: Literal["training", "evaluation"]
    extent: tuple[int, int, int]
    world_identity: str | None = None
    world: ScenarioWorld = Field(exclude=True)
    candidates: Any = Field(default=None, exclude=True)
    action_mode: str
    action_offsets: tuple[tuple[int, int, int], ...]
    previous_scenario: dict[str, Any] | None = None
    curriculum: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Start and ordered goal route selected for one episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: tuple[int, int, int]
    goal: tuple[int, int, int] | None = None
    route: tuple[tuple[int, int, int], ...] = ()
    scenario_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_route(self) -> ScenarioResult:
        """Require exactly one non-empty representation of the goal route."""
        if self.goal is None and not self.route:
            raise ValueError("scenario result requires goal or route")
        if self.goal is not None and self.route:
            raise ValueError("scenario result cannot define both goal and route")
        return self

    @property
    def waypoints(self) -> tuple[tuple[int, int, int], ...]:
        """Return the normalized ordered goal sequence."""
        return self.route or (self.goal,)  # type: ignore[return-value]


ScenarioFunction = Callable[[ScenarioContext], ScenarioResult]


class ScenarioProvider(BaseModel):
    """Validated Python or native scenario function."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    source_path: Path
    source_sha256: str
    generate: ScenarioFunction | None = Field(default=None, exclude=True)
    native_abi: Literal[1, 2] | None = None


def available_python_scenario_names(
    source_path: Path | None,
    candidates: tuple[str, ...],
) -> tuple[str, ...]:
    """Return selected names implemented by one Python scenario module."""
    if source_path is None:
        return ()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"_theseo_anysearch_scenarios_probe_{digest[:16]}", source_path
    )
    if spec is None or spec.loader is None:
        raise CustomScenarioError(f"Cannot import custom scenarios from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    available: list[str] = []
    for name in candidates:
        function = getattr(module, name, None)
        if function is None:
            continue
        if not callable(function):
            raise CustomScenarioError(f"{source_path}: {name} must be callable")
        if len(inspect.signature(function).parameters) != 1:
            raise CustomScenarioError(
                f"{source_path}: {name} must accept exactly one argument"
            )
        available.append(name)
    return tuple(available)


def discover_scenario_source(
    config_path: Path | None, provider_name: str | None
) -> Path | None:
    """Discover the conventional sibling ``scenarios.py`` module."""
    if config_path is None:
        return None
    if provider_name is None:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        owners = [raw.get("env") or {}, raw.get("evaluation") or {}]
        for stage in (raw.get("staging") or {}).get("stages") or []:
            owners.extend((stage.get("env") or {}, stage.get("evaluation") or {}))
        if not any((owner.get("scenarios") or {}).get("provider") for owner in owners):
            return None
    source = config_path.with_name("scenarios.py")
    return source if source.is_file() else None


def copy_scenario_source(
    config_path: Path | None, destination: Path, provider_name: str | None
) -> Path | None:
    """Archive the exact Python scenario source used by a run."""
    source = discover_scenario_source(config_path, provider_name)
    if source is None:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination.joinpath("scenarios.py")
    shutil.copy2(source, target)
    return target


def load_scenario_provider(
    source_path: Path | None, provider_name: str | None
) -> ScenarioProvider | None:
    """Load one named function from ``scenarios.py`` without fallbacks."""
    if source_path is None or provider_name is None:
        return None
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"_theseo_anysearch_scenarios_{digest[:16]}", source_path
    )
    if spec is None or spec.loader is None:
        raise CustomScenarioError(f"Cannot import custom scenarios from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, provider_name, None)
    if not callable(function):
        raise CustomScenarioError(
            f"{source_path} must define callable {provider_name}(context)"
        )
    if len(inspect.signature(function).parameters) != 1:
        raise CustomScenarioError(
            f"{source_path}: {provider_name} must accept exactly one argument"
        )

    def generate(context: ScenarioContext) -> ScenarioResult:
        return ScenarioResult.model_validate(function(context))

    return ScenarioProvider(
        name=provider_name,
        source_path=source_path,
        source_sha256=digest,
        generate=generate,
    )


def load_native_scenario_provider(
    library_path: Path, provider_name: str
) -> ScenarioProvider:
    """Load a macro-generated Rust scenario function through the stable JSON ABI."""
    library = ctypes.CDLL(str(library_path))
    v2_symbol = f"anysearch_scenario_{provider_name}_v2"
    if getattr(library, v2_symbol, None) is not None:
        digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
        return ScenarioProvider(
            name=provider_name,
            source_path=library_path,
            source_sha256=digest,
            native_abi=2,
        )
    symbol = f"anysearch_scenario_{provider_name}_v1"
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    function.restype = ctypes.c_int32

    def generate(context: ScenarioContext) -> ScenarioResult:
        payload = context.model_dump_json().encode()
        output = ctypes.create_string_buffer(65536)
        length = ctypes.c_size_t()
        status = int(
            function(payload, len(payload), output, len(output), ctypes.byref(length))
        )
        if status != 0:
            raise CustomScenarioError(
                f"native scenario {provider_name!r} failed with status {status}"
            )
        return ScenarioResult.model_validate_json(output.raw[: length.value])

    digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
    return ScenarioProvider(
        name=provider_name,
        source_path=library_path,
        source_sha256=digest,
        generate=generate,
        native_abi=1,
    )


def validate_scenario(
    result: ScenarioResult,
    *,
    extent: tuple[int, int, int],
    world: ScenarioWorld,
) -> ScenarioResult:
    """Reject out-of-grid, occupied, or degenerate scenario coordinates."""
    coordinates = (result.start, *result.waypoints)
    for coordinate in coordinates:
        if not all(1 <= coordinate[axis] <= extent[axis] for axis in range(3)):
            raise CustomScenarioError(
                f"scenario {result.scenario_id!r} coordinate {coordinate} is outside grid"
            )
        if world.occupied(coordinate):
            raise CustomScenarioError(
                f"scenario {result.scenario_id!r} coordinate {coordinate} is occupied"
            )
    if result.start == result.waypoints[0]:
        raise CustomScenarioError(
            f"scenario {result.scenario_id!r} start equals its first goal"
        )
    return result
