"""Validation tests for public explanation documents."""

import numpy as np
import pytest
from gymnasium import spaces
from pydantic import ValidationError

from theseo_anysearch.rllib.explain.requests import ExplanationRequestFile
from theseo_anysearch.rllib.explain.scenarios import validate_observation


def test_request_requires_exactly_one_source() -> None:
    """Missing and conflicting sources are never silently selected."""

    with pytest.raises(ValidationError, match="exactly one"):
        ExplanationRequestFile.model_validate({"source": {}})
    with pytest.raises(ValidationError, match="exactly one"):
        ExplanationRequestFile.model_validate(
            {"source": {"trace": "best", "scenario": "scenario.yaml"}}
        )


def test_request_rejects_unknown_configuration() -> None:
    """Typographical errors must remain visible to users."""

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExplanationRequestFile.model_validate(
            {"source": {"trace": "best"}, "explanation": {"methd": "occlusion"}}
        )


def test_fictional_observation_is_validated_exactly() -> None:
    """Observation keys, shapes, and declared bounds are authoritative."""

    observation_space = spaces.Dict(
        {"goal_distance": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)}
    )
    result = validate_observation({"goal_distance": [0.5]}, observation_space)
    assert result["goal_distance"].tolist() == [0.5]
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_observation({"unknown": [0.5]}, observation_space)
    with pytest.raises(ValueError, match="outside its declared bounds"):
        validate_observation({"goal_distance": [2.0]}, observation_space)
