"""Contract tests for composable episode lifecycle rules."""

import pytest

from theseo_anysearch.environments.lifecycle import (
    LifecycleContext,
    LifecycleDecision,
    build_lifecycle_rules,
    evaluate_lifecycle,
    register_lifecycle_rule,
)


def context(**overrides):
    values = {
        "step": 1, "action": 0, "action_index": 0, "cursor": (1, 2, 3),
        "goal": (2, 2, 3), "goal_distance": 1.0, "collision": False,
        "invalid_action": False, "native_success": False,
        "native_terminated": False, "native_truncated": False,
        "native_reason": "in_progress", "route_complete": True,
    }
    values.update(overrides)
    return LifecycleContext(**values)


def test_native_rule_preserves_success_and_truncation_contract():
    rules = build_lifecycle_rules([{"name": "native"}])
    success = evaluate_lifecycle(
        rules, context(native_success=True, native_terminated=True, native_reason="success")
    )
    truncated = evaluate_lifecycle(
        rules, context(native_truncated=True, native_reason="step_limit")
    )
    assert (success.success, success.terminated, success.truncated, success.reason) == (
        True, True, False, "success"
    )
    assert (truncated.failure, truncated.terminated, truncated.truncated) == (
        False, False, True
    )


def test_ordered_rules_compose_and_terminal_outcome_wins_over_truncation():
    register_lifecycle_rule(
        "test_budget",
        lambda parameters: lambda value: LifecycleDecision(
            truncated=value.step >= parameters["steps"],
            reason="custom_budget",
            diagnostics={"budget": parameters["steps"]},
        ),
        replace=True,
    )
    register_lifecycle_rule(
        "test_success",
        lambda parameters: lambda value: LifecycleDecision(
            success=value.cursor == tuple(parameters["target"]),
            reason="custom_success",
            diagnostics={"matched_target": True},
        ),
        replace=True,
    )
    rules = build_lifecycle_rules([
        {"name": "native"},
        {"name": "test_budget", "parameters": {"steps": 1}},
        {"name": "test_success", "parameters": {"target": [1, 2, 3]}},
    ])
    outcome = evaluate_lifecycle(rules, context())
    assert outcome.success is True
    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.reason == "custom_success"
    assert outcome.diagnostics == {"budget": 1, "matched_target": True}


def test_conflicting_results_and_diagnostic_keys_fail_loudly():
    with pytest.raises(ValueError, match="both success and failure"):
        evaluate_lifecycle((
            lambda value: LifecycleDecision(success=True),
            lambda value: LifecycleDecision(failure=True),
        ), context())
    with pytest.raises(ValueError, match="diagnostic keys"):
        evaluate_lifecycle((
            lambda value: LifecycleDecision(diagnostics={"shared": 1}),
            lambda value: LifecycleDecision(diagnostics={"shared": 2}),
        ), context())


def test_same_context_and_rules_produce_same_decision():
    rules = build_lifecycle_rules([{"name": "native"}])
    transition = context(native_truncated=True, native_reason="step_limit")
    assert evaluate_lifecycle(rules, transition) == evaluate_lifecycle(rules, transition)
