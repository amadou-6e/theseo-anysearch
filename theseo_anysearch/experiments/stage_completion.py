"""Evaluation of composable staged-training completion conditions."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from theseo_anysearch.experiments.models import StageCompletionConfig
from theseo_anysearch.rllib.trainer.results import TrainResult


_COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "gte": lambda value, threshold: value >= threshold,
    "gt": lambda value, threshold: value > threshold,
    "lte": lambda value, threshold: value <= threshold,
    "lt": lambda value, threshold: value < threshold,
    "eq": lambda value, threshold: value == threshold,
}



def _resolve_performance_metric(
    metric: str,
    result: TrainResult,
    metrics: dict[str, float],
) -> float | None:
    """Look up a performance-condition metric, falling back to TrainResult fields.

    standard_metrics() (see #195) now only emits canonically-namespaced
    metrics (``eval/task/...``, ``train/task/...``), so it no longer
    unconditionally includes every legacy top-level TrainResult field (e.g.
    ``evaluation_success_rate``). Existing experiment YAML configs may still
    reference those field names directly, and the fields themselves are
    still populated, so fall back to reading them straight off ``result``.
    """
    if metric in metrics:
        return metrics[metric]
    value = getattr(result, metric, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_callable(reference: str) -> Callable[[dict[str, Any]], bool]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("python completion callable must use 'module.path:function'")
    candidate: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        candidate = getattr(candidate, part)
    if not callable(candidate):
        raise TypeError(f"python completion target is not callable: {reference}")
    return candidate


class StageCompletionController:
    """Evaluate a condition tree and retain its JSON-serializable state."""

    def __init__(
        self,
        config: StageCompletionConfig,
        *,
        stage_start_iteration: int,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.stage_start_iteration = stage_start_iteration
        self.state = state or {}
        self.completed = False
        self.halted = False
        self.exhausted = False
        self.reason = ""

    def evaluate(self, result: TrainResult) -> bool:
        local_iteration = result.iteration - self.stage_start_iteration
        metrics = result.standard_metrics()
        self.completed = self._evaluate(
            self.config, result, metrics, local_iteration, "root"
        )
        if self.completed:
            self.reason = f"condition:{self.config.type}"
        elif (
            self.config.max_iterations is not None
            and local_iteration >= self.config.max_iterations
        ):
            self.exhausted = True
            policy = self.config.on_max_iterations
            self.reason = f"max_iterations={self.config.max_iterations}:{policy}"
            if policy == "advance":
                self.completed = True
            elif policy == "stop":
                self.halted = True
            else:
                raise RuntimeError(
                    "stage exhausted max_iterations without satisfying its "
                    "completion condition"
                )
        return self.completed

    def _evaluate(
        self,
        config: StageCompletionConfig,
        result: TrainResult,
        metrics: dict[str, float],
        local_iteration: int,
        path: str,
    ) -> bool:
        if config.type == "iterations":
            matched = local_iteration >= int(config.iterations or 0)
        elif config.type == "performance":
            value = _resolve_performance_metric(config.metric, result, metrics)
            if value is None:
                available = ", ".join(sorted(metrics))
                raise KeyError(
                    f"unknown completion metric '{config.metric}'; available: {available}"
                )
            matched_now = _COMPARISONS[config.comparison](
                value, float(config.threshold)
            )
            consecutive_key = f"{path}.consecutive"
            consecutive = int(self.state.get(consecutive_key, 0))
            consecutive = consecutive + 1 if matched_now else 0
            self.state[consecutive_key] = consecutive
            matched = consecutive >= config.consecutive_iterations
        elif config.type in {"all", "any"}:
            values = [
                self._evaluate(
                    child, result, metrics, local_iteration, f"{path}.{index}"
                )
                for index, child in enumerate(config.conditions)
            ]
            matched = all(values) if config.type == "all" else any(values)
        elif config.type == "not":
            matched = not self._evaluate(
                config.condition, result, metrics, local_iteration, f"{path}.not"
            )
        else:
            extension_state = self.state.setdefault(f"{path}.python", {})
            matched = bool(_load_callable(config.callable or "")({
                "result": result,
                "metrics": metrics,
                "stage_iteration": local_iteration,
                "state": extension_state,
                "parameters": config.parameters,
            }))
        return matched
