"""Scenario settings tests."""

from theseo_anysearch.settings.environment import EnvConfig
from theseo_anysearch.settings.evaluation import EvaluationConfig


def test_env_scenario_shorthand_reaches_runtime() -> None:
    config = EnvConfig(scenarios={"provider": "adjacent_goal"})

    assert config.to_runtime_dict()["scenario_provider"] == "adjacent_goal"


def test_evaluation_scenario_parameters_are_typed() -> None:
    config = EvaluationConfig(
        scenarios={
            "provider": {
                "name": "suite",
                "parameters": {"count": 26},
            }
        }
    )

    assert config.scenarios.provider is not None
    assert config.scenarios.provider.parameters == {"count": 26}
