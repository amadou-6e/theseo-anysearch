"""Weighted sampling strategies for visited waypoint curriculum stages."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.models import WaypointTrainingSamplingConfig

Waypoint = tuple[int, int, int]


class StageSamplingStage(BaseModel):
    """Read-only facts available to a stage sampling function."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    index: int = Field(ge=0)
    start: Waypoint
    goal: Waypoint
    age: int = Field(ge=0)
    is_latest: bool
    evaluation_attempts: int = Field(default=0, ge=0)
    evaluation_successes: int = Field(default=0, ge=0)
    evaluation_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class StageSamplingContext(BaseModel):
    """Complete input passed to custom curriculum stage samplers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stages: tuple[StageSamplingStage, ...]


StageSamplingFunction = Callable[
    [StageSamplingContext], Mapping[int, float] | Sequence[float]
]
_CUSTOM_SAMPLERS: dict[str, StageSamplingFunction] = {}
_CUSTOM_MODULE_LOADED = False


def stage_sampling(function: StageSamplingFunction) -> StageSamplingFunction:
    """Register a custom sampler under its Python function name."""
    if function.__name__ in _CUSTOM_SAMPLERS:
        raise ValueError(
            f"stage sampling function {function.__name__!r} is already registered"
        )
    _CUSTOM_SAMPLERS[function.__name__] = function
    return function


def stage_probabilities(
    context: StageSamplingContext,
    config: WaypointTrainingSamplingConfig,
) -> list[float]:
    """Calculate and normalize probabilities for every visited stage."""
    if not context.stages:
        raise ValueError("stage sampling requires at least one visited stage")

    strategy = config.strategy
    if strategy == "legacy":
        weights = _legacy_weights(len(context.stages), config)
    elif strategy == "uniform":
        weights = [1.0] * len(context.stages)
    elif strategy == "latest_multiplier":
        weights = [1.0] * len(context.stages)
        weights[-1] = config.latest_multiplier
    elif strategy == "recency":
        weights = [
            max(config.minimum_weight, config.recency_decay**stage.age)
            for stage in context.stages
        ]
    elif strategy == "inverse_success":
        weights = [
            max(
                config.minimum_weight,
                (
                    1.0
                    - (
                        stage.evaluation_success_rate
                        if stage.evaluation_success_rate is not None
                        else config.unevaluated_success_rate
                    )
                )
                ** config.power,
            )
            for stage in context.stages
        ]
    else:
        function = _custom_sampler(strategy)
        returned = function(context)
        weights = _coerce_custom_weights(returned, len(context.stages))
    return _normalize(weights)


def _legacy_weights(
    count: int,
    config: WaypointTrainingSamplingConfig,
) -> list[float]:
    if count == 1:
        return [1.0]
    retained = config.retained_stage_probability / (count - 1)
    return [retained] * (count - 1) + [config.current_stage_probability]


def _normalize(weights: Sequence[float]) -> list[float]:
    values = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in values):
        raise ValueError("stage sampling weights must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("at least one stage sampling weight must be positive")
    return [weight / total for weight in values]


def _coerce_custom_weights(
    returned: Mapping[int, float] | Sequence[float],
    count: int,
) -> list[float]:
    if isinstance(returned, Mapping):
        expected = set(range(count))
        if set(returned) != expected:
            raise ValueError(
                "custom stage sampler must return one weight per stage index"
            )
        return [float(returned[index]) for index in range(count)]
    values = [float(value) for value in returned]
    if len(values) != count:
        raise ValueError("custom stage sampler must return one weight per stage")
    return values


def _custom_sampler(name: str) -> StageSamplingFunction:
    _load_custom_module()
    try:
        return _CUSTOM_SAMPLERS[name]
    except KeyError as error:
        available = ", ".join(sorted(_CUSTOM_SAMPLERS)) or "none"
        raise ValueError(
            f"unknown stage sampling strategy {name!r}; registered custom strategies: {available}"
        ) from error


def _load_custom_module() -> None:
    global _CUSTOM_MODULE_LOADED
    if _CUSTOM_MODULE_LOADED:
        return
    _CUSTOM_MODULE_LOADED = True
    path = Path("curriculum_sampling.py")
    if not path.is_file():
        return
    spec = importlib.util.spec_from_file_location("anysearch_curriculum_sampling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load custom curriculum sampling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
