"""Settings tests for heterogeneous multi-agent environments."""

import pytest
from pydantic import ValidationError

from theseo_anysearch.settings import EnvConfig


def agents() -> list[dict]:
    """Return two distinct agent configurations."""
    return [
        {"id": "hunted", "action": {"behavior": "cursor_navigation"}},
        {
            "id": "hunter",
            "action": {
                "predicates": ["valid_action", {"name": "double_step_in_bounds"}],
                "outcomes": [{"name": "double_step"}],
            },
        },
    ]


def test_agents_derive_agent_count_and_runtime_pipelines() -> None:
    config = EnvConfig(agents=agents())
    runtime = config.to_runtime_dict()
    assert config.agent_count == 2
    assert [agent["id"] for agent in runtime["agents"]] == ["hunted", "hunter"]
    assert runtime["agents"][1]["action_outcomes"][0]["name"] == "double_step"


def test_capture_task_must_reference_configured_agents() -> None:
    with pytest.raises(ValidationError, match="hunter.*configured agent"):
        EnvConfig(
            agents=agents(),
            hunter_and_hunted={"hunter": "missing", "hunted": "hunted"},
        )


def test_explicit_agent_count_must_match_agents() -> None:
    with pytest.raises(ValidationError, match="agent_count"):
        EnvConfig(agent_count=3, agents=agents())
