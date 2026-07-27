"""Cross-field preflight tests for imitation configuration."""

import pytest
from pydantic import ValidationError

from theseo_anysearch.experiments.models import ExperimentConfig


def _experiment(algorithm: str, agent_count: int = 1) -> dict:
    return {
        "experiment": {"name": "imitation-validation"},
        "env": {"agent_count": agent_count},
        "training": {"algorithm": algorithm},
        "imitation": {
            "enabled": True,
            "collection": {"episodes": 2, "max_attempts": 2},
        },
    }


def test_imitation_rejects_non_ppo_algorithm_before_runtime():
    with pytest.raises(ValidationError, match="supports PPO only"):
        ExperimentConfig.model_validate(_experiment("dqn"))


def test_imitation_rejects_multi_agent_environment_before_runtime():
    with pytest.raises(ValidationError, match="requires env.agent_count: 1"):
        ExperimentConfig.model_validate(_experiment("ppo", agent_count=2))


def test_disabled_imitation_remains_backward_compatible():
    raw = _experiment("ppo")
    raw["imitation"] = {"enabled": False}

    config = ExperimentConfig.model_validate(raw)

    assert config.imitation.enabled is False
