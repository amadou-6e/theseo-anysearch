"""Unit tests for composable stage completion conditions."""

import sys
from types import ModuleType

from theseo_anysearch.experiments.models import StageCompletionConfig
from theseo_anysearch.experiments.stage_completion import StageCompletionController
from theseo_anysearch.rllib.trainer.results import TrainResult


def _result(iteration: int, success: float = 0.0) -> TrainResult:
    return TrainResult(
        iteration=iteration,
        episode_reward_mean=success,
        episode_len_mean=1.0,
        episodes_total=iteration,
        elapsed_s=0.1,
        evaluation_success_rate=success,
    )


def test_iteration_completion_uses_stage_local_iteration():
    config = StageCompletionConfig(type="iterations", iterations=3)
    controller = StageCompletionController(config, stage_start_iteration=10)

    assert not controller.evaluate(_result(12))
    assert controller.evaluate(_result(13))


def test_performance_completion_requires_consecutive_matches():
    config = StageCompletionConfig(
        type="performance",
        metric="evaluation_success_rate",
        threshold=0.9,
        consecutive_iterations=2,
        max_iterations=10,
    )
    controller = StageCompletionController(config, stage_start_iteration=0)

    assert not controller.evaluate(_result(1, 0.95))
    assert not controller.evaluate(_result(2, 0.5))
    assert not controller.evaluate(_result(3, 0.95))
    assert controller.evaluate(_result(4, 0.95))


def test_nested_any_condition_composes_performance_and_iteration():
    config = StageCompletionConfig.model_validate({
        "type": "any",
        "conditions": [
            {
                "type": "performance",
                "metric": "evaluation_success_rate",
                "threshold": 0.9,
            },
            {"type": "iterations", "iterations": 3},
        ],
    })
    controller = StageCompletionController(config, stage_start_iteration=0)

    assert not controller.evaluate(_result(2, 0.1))
    assert controller.evaluate(_result(3, 0.1))


def test_completion_state_can_be_restored():
    config = StageCompletionConfig(
        type="performance",
        metric="evaluation_success_rate",
        threshold=0.9,
        consecutive_iterations=2,
        max_iterations=5,
    )
    first = StageCompletionController(config, stage_start_iteration=4)
    assert not first.evaluate(_result(5, 1.0))

    resumed = StageCompletionController(
        config, stage_start_iteration=4, state=first.state
    )
    assert resumed.evaluate(_result(6, 1.0))


def test_python_condition_receives_parameters_and_persisted_state(monkeypatch):
    extension = ModuleType("test_stage_extension")

    def complete(context):
        context["state"]["calls"] = context["state"].get("calls", 0) + 1
        return context["state"]["calls"] >= context["parameters"]["calls"]

    extension.complete = complete
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    config = StageCompletionConfig(
        type="python",
        callable="test_stage_extension:complete",
        parameters={"calls": 2},
        max_iterations=5,
    )
    controller = StageCompletionController(config, stage_start_iteration=0)

    assert not controller.evaluate(_result(1))
    assert controller.evaluate(_result(2))
    assert controller.state["root.python"]["calls"] == 2
