"""Convention-based custom metric discovery and execution."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

MetricScope = Literal["evaluation", "training"]
MetricFunction = Callable[[Any], Mapping[str, float]]


class CustomMetricError(ValueError):
    """Raised when a custom metric module violates the metric contract."""


class EvaluationContext(BaseModel):
    """Inputs provided to an evaluation metric function."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    iteration: int
    episodes: tuple[Any, ...]
    standard_metrics: dict[str, float]
    env_config: dict[str, Any]
    final_infos: tuple[dict[str, Any], ...]


class TrainingContext(BaseModel):
    """Inputs provided to a training metric function."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    iteration: int
    standard_metrics: dict[str, float]
    rllib_result: dict[str, Any]
    environment_steps_total: int
    duration_s: float
    env_config: dict[str, Any]


class MetricProvider(BaseModel):
    """A validated metric function loaded from a discovered source module."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    scope: MetricScope
    source_path: Path
    source_sha256: str
    compute_metrics: MetricFunction = Field(exclude=True)


class MetricProviders(BaseModel):
    """Optional training and evaluation providers for one experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    evaluation: MetricProvider | None = None
    training: MetricProvider | None = None
    scopes: ClassVar[tuple[MetricScope, ...]] = ("evaluation", "training")

    def provider(self, scope: MetricScope) -> MetricProvider | None:
        return getattr(self, scope)


def _candidate_paths(config_path: Path, scope: MetricScope) -> tuple[Path, Path]:
    return (
        config_path.with_name(f"{scope}_metrics.{config_path.stem}.py"),
        config_path.with_name(f"{scope}_metrics.py"),
    )


def discover_metric_sources(config_path: Path | None) -> dict[MetricScope, Path]:
    """Prefer experiment-specific metric modules, then shared modules."""
    if config_path is None:
        return {}
    discovered: dict[MetricScope, Path] = {}
    for scope in MetricProviders.scopes:
        specific, shared = _candidate_paths(config_path, scope)
        if specific.is_file():
            discovered[scope] = specific
        elif shared.is_file():
            discovered[scope] = shared
    return discovered


def copy_metric_sources(
    config_path: Path | None,
    destination: Path,
) -> dict[MetricScope, Path]:
    """Copy discovered sources beside the run-local experiment YAML."""
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[MetricScope, Path] = {}
    for scope, source in discover_metric_sources(config_path).items():
        target = destination.joinpath(f"{scope}_metrics.py")
        shutil.copy2(source, target)
        copied[scope] = target
    return copied


def _load_provider(source_path: Path, scope: MetricScope) -> MetricProvider:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"_theseo_anysearch_{scope}_metrics_{digest[:16]}", source_path
    )
    if spec is None or spec.loader is None:
        raise CustomMetricError(f"Cannot import custom metrics from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "compute_metrics", None)
    if not callable(function):
        raise CustomMetricError(
            f"{source_path} must define callable compute_metrics(context)"
        )
    if len(inspect.signature(function).parameters) != 1:
        raise CustomMetricError(
            f"{source_path}: compute_metrics must accept exactly one argument"
        )
    return MetricProvider(
        scope=scope,
        source_path=source_path,
        source_sha256=digest,
        compute_metrics=function,
    )


def load_metric_providers(config_path: Path | None) -> MetricProviders:
    """Discover, import, and validate both optional metric providers."""
    sources = discover_metric_sources(config_path)
    return MetricProviders(
        evaluation=(
            _load_provider(sources["evaluation"], "evaluation")
            if "evaluation" in sources
            else None
        ),
        training=(
            _load_provider(sources["training"], "training")
            if "training" in sources
            else None
        ),
    )


def compute_custom_metrics(
    provider: MetricProvider | None,
    context: EvaluationContext | TrainingContext,
    *,
    reserved_names: set[str],
) -> dict[str, float]:
    """Execute a provider, validate its result, and prefix metric names."""
    if provider is None:
        return {}
    raw = provider.compute_metrics(context)
    if not isinstance(raw, Mapping):
        raise CustomMetricError(
            f"{provider.source_path}: compute_metrics must return a mapping"
        )
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise CustomMetricError(
                f"{provider.source_path}: metric names must be Python identifiers"
            )
        full_name = f"{provider.scope}_{name}"
        if full_name in reserved_names or full_name in metrics:
            raise CustomMetricError(
                f"{provider.source_path}: metric {full_name!r} is reserved or duplicated"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CustomMetricError(
                f"{provider.source_path}: metric {name!r} must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise CustomMetricError(
                f"{provider.source_path}: metric {name!r} must be finite"
            )
        metrics[full_name] = numeric
    return metrics


def write_metric_manifest(
    providers: MetricProviders,
    destination: Path,
) -> Path | None:
    """Record the archived provider source names and hashes."""
    entries = {
        scope: {
            "source": provider.source_path.name,
            "sha256": provider.source_sha256,
        }
        for scope in MetricProviders.scopes
        if (provider := providers.provider(scope)) is not None
    }
    if not entries:
        return None
    path = destination.joinpath("custom_metrics.json")
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path
