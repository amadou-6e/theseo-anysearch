"""Tests for imitation provider config shape."""

import pytest
from pydantic import ValidationError

from theseo_anysearch.imitation.models import GenerationConfig, ImitationConfig, SamplingConfig


def test_generation_provider_accepts_scalar_shorthand():
    config = GenerationConfig(provider="astar")
    assert config.provider.name == "astar"
    assert config.provider.parameters == {}


def test_sampling_provider_accepts_scalar_shorthand():
    config = SamplingConfig(provider="uniform_episode")
    assert config.provider.name == "uniform_episode"


def test_generation_provider_accepts_name_and_parameters_block():
    config = GenerationConfig(provider={"name": "weighted_astar", "parameters": {"weight": 2.0}})
    assert config.provider.name == "weighted_astar"
    assert config.provider.parameters == {"weight": 2.0}


def test_imitation_config_defaults_to_astar_generation_and_uniform_transition_sampling():
    config = ImitationConfig()
    assert config.generation.provider.name == "astar"
    assert config.sampling.provider.name == "uniform_transition"


def test_imitation_config_rejects_legacy_teacher_key():
    with pytest.raises(ValidationError, match="teacher"):
        ImitationConfig(teacher={"type": "astar"})


def test_generation_attempt_budget_must_cover_requested_episodes():
    with pytest.raises(ValidationError, match="max_attempts must be at least episodes"):
        GenerationConfig(episodes=5, max_attempts=4)


def test_collection_no_longer_accepts_episodes_field():
    from theseo_anysearch.imitation.models import DemonstrationCollectionConfig

    with pytest.raises(ValidationError):
        DemonstrationCollectionConfig(episodes=5)
