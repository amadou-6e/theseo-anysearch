"""Unit tests for experiment-local scenario providers."""

from pathlib import Path

import pytest

from theseo_anysearch.experiments.custom_scenarios import (
    CustomScenarioError,
    ScenarioContext,
    ScenarioResult,
    available_python_scenario_names,
    load_scenario_provider,
    validate_scenario,
)
from theseo_anysearch.experiments.loader import load_experiment


class EmptyWorld:
    extent = (8, 8, 8)
    identity = None

    def occupied(self, coordinate):
        return False

    def occupied_in_region(self, minimum, maximum_exclusive):
        return ()


def context(**updates) -> ScenarioContext:
    values = {
        "seed": 42,
        "episode_index": 0,
        "scope": "evaluation",
        "extent": (8, 8, 8),
        "world": EmptyWorld(),
        "action_mode": "discrete_26",
        "action_offsets": ((1, 0, 0),),
    }
    values.update(updates)
    return ScenarioContext(**values)


def test_loads_named_python_scenario(tmp_path: Path) -> None:
    source = tmp_path.joinpath("scenarios.py")
    source.write_text(
        "def adjacent(context):\n"
        "    return {'start': (2, 2, 2), 'goal': (3, 2, 2), "
        "'scenario_id': f'episode-{context.episode_index}'}\n",
        encoding="utf-8",
    )

    provider = load_scenario_provider(source, "adjacent")

    assert provider is not None
    assert provider.generate(context(episode_index=7)).scenario_id == "episode-7"


def test_discovers_only_selected_python_implementations(tmp_path: Path) -> None:
    source = tmp_path.joinpath("scenarios.py")
    source.write_text(
        "def python_only(context):\n    return {}\n",
        encoding="utf-8",
    )

    assert available_python_scenario_names(source, ("python_only", "rust_only")) == (
        "python_only",
    )


def test_tune_materializes_python_scenarios_for_each_trial(tmp_path: Path) -> None:
    from theseo_anysearch.experiments.tune_runner import (
        _write_trial_extension_sources,
    )

    _write_trial_extension_sources(
        tmp_path,
        metric_source_contents=None,
        reward_source_content=None,
        scenario_source_content="def adjacent(context):\n    return {}\n",
        generation_source_content=None,
    )

    assert (
        tmp_path.joinpath("scenarios.py")
        .read_text(encoding="utf-8")
        .startswith("def adjacent")
    )


def test_rejects_occupied_scenario_coordinate() -> None:
    scenario = ScenarioResult(start=(2, 2, 2), goal=(3, 2, 2), scenario_id="occupied")

    class OccupiedWorld(EmptyWorld):
        def occupied(self, coordinate):
            return coordinate == (3, 2, 2)

    with pytest.raises(CustomScenarioError, match="occupied"):
        validate_scenario(scenario, extent=(8, 8, 8), world=OccupiedWorld())


def test_result_requires_goal_or_route() -> None:
    with pytest.raises(ValueError, match="requires goal or route"):
        ScenarioResult(start=(2, 2, 2), scenario_id="empty")


def test_adjacent_scenario_showcase_resolves_training_and_evaluation_providers() -> (
    None
):
    config_path = Path(
        "usage", "experiments", "showcase", "scenario_extensions", "experiment.yaml"
    )
    experiment = load_experiment(config_path)

    assert experiment.staging is None
    assert experiment.env.max_steps == 2
    assert experiment.env.scenarios.provider is not None
    assert experiment.env.scenarios.provider.name == "adjacent_goal_python"
    assert experiment.evaluation.episodes == 26
    assert experiment.evaluation.scenarios.provider is not None
    assert experiment.evaluation.scenarios.provider.name == "adjacent_goal_rust"


def test_adjacent_scenario_evaluation_covers_every_action_direction() -> None:
    source = Path(
        "usage",
        "experiments",
        "showcase",
        "scenario_extensions",
        "scenarios.py",
    )
    provider = load_scenario_provider(source, "adjacent_goal_python")
    offsets = tuple(
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    )

    scenarios = [
        provider.generate(
            context(
                seed=142 + index,
                extent=(32, 32, 32),
                action_offsets=offsets,
                parameters={"seed_base": 142},
            )
        )
        for index in range(26)
    ]

    assert {scenario.goal for scenario in scenarios} == {
        tuple(16 + coordinate for coordinate in offset) for offset in offsets
    }
