from __future__ import annotations

import pytest

from theseo_anysearch import models as compatibility_models
from theseo_anysearch.settings import (
    ActionConfig,
    AlgorithmConfig,
    AnyscaleConfig,
    EnvConfig,
    EvaluationConfig,
    GeometryConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
    RewardSelector,
    Settings,
    TrainingConfig,
    TrainingEarlyStopConfig,
)


def test_reward_provider_accepts_structured_and_shorthand_selection():
    structured = RewardConfig.model_validate({
        "provider": {"name": "segment_goal", "parameters": {"minimum": 1.0}}
    })
    shorthand = RewardConfig.model_validate({"provider": "segment_goal"})

    assert structured.provider == RewardSelector(
        name="segment_goal", parameters={"minimum": 1.0}
    )
    assert shorthand.provider == RewardSelector(name="segment_goal")


def test_legacy_custom_reward_key_and_property_remain_compatible():
    rewards = RewardConfig.model_validate({"custom": "legacy_reward"})

    assert rewards.provider == RewardSelector(name="legacy_reward")
    assert rewards.custom is rewards.provider
    assert "provider" in rewards.model_dump()
    assert "custom" not in rewards.model_dump()


def test_models_module_reexports_settings_contracts():
    assert compatibility_models.EnvConfig is EnvConfig
    assert compatibility_models.RewardSelector is RewardSelector
    assert compatibility_models.Settings is Settings


def test_every_shared_yaml_field_has_a_description():
    settings_models = (
        GeometryConfig,
        ObservationConfig,
        ActionConfig,
        RewardSelector,
        RewardConfig,
        EnvConfig,
        TrainingEarlyStopConfig,
        TrainingConfig,
        EvaluationConfig,
        AnyscaleConfig,
        AlgorithmConfig,
        ModelConfig,
        Settings,
    )

    missing = {
        model.__name__: [name for name, field in model.model_fields.items() if not field.description]
        for model in settings_models
    }
    assert {name: fields for name, fields in missing.items() if fields} == {}

def test_provider_and_legacy_custom_cannot_be_mixed():
    with pytest.raises(ValueError, match="cannot define both"):
        RewardConfig.model_validate({"provider": "one", "custom": "two"})