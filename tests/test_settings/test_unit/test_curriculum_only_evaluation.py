import pytest
from pydantic import ValidationError

from theseo_anysearch.settings.evaluation import EvaluationConfig


def test_regular_evaluation_is_enabled_by_default():
    assert EvaluationConfig().enabled


def test_curriculum_evaluation_remains_enabled_independently():
    config = EvaluationConfig(enabled=False, waypoint_curriculum={
        "enabled": True, "frequency": 5, "episodes": 3,
    })
    assert not config.enabled
    assert config.waypoint_curriculum.enabled
    assert config.waypoint_curriculum.frequency == 5
    assert config.waypoint_curriculum.episodes == 3


def test_disabled_evaluation_rejects_parallel_scheduling():
    with pytest.raises(ValidationError, match="disabled regular evaluation"):
        EvaluationConfig(enabled=False, parallel_to_training=True, num_env_runners=1)
