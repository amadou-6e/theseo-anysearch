"""Tests for explainability feature schemas."""

from __future__ import annotations

import numpy as np
import pytest

from theseo_anysearch.rllib.explain.features import FeatureSchema


def radial_observation() -> dict[str, np.ndarray]:
    """Build a complete radial observation fixture."""

    return {
        "steps_remaining": np.asarray([1.0], dtype=np.float32),
        "goal_distance": np.asarray([0.2], dtype=np.float32),
        "goal_direction": np.asarray([0.1, 0.0, 0.2], dtype=np.float32),
        "cursor_pos": np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        "ray_hits": np.linspace(0.0, 1.0, 26, dtype=np.float32),
        "ray_hit_types": np.linspace(0.0, 1.0, 26, dtype=np.float32),
    }


def test_feature_schema_flattens_and_unflattens_radial_observation():
    observation = radial_observation()
    schema = FeatureSchema.from_observation(observation)

    flat = schema.flatten(observation)
    restored = schema.unflatten(flat)

    assert flat.shape == (60,)
    for name, value in observation.items():
        assert restored[name] == pytest.approx(value)


def test_feature_schema_names_action_aligned_ray_features():
    schema = FeatureSchema.from_observation(radial_observation())

    names = schema.feature_names()

    assert names[schema.action_feature_index("ray_hits", 21)] == "ray_hits[21](dx=1,dy=0,dz=0)"
    assert (
        names[schema.action_feature_index("ray_hit_types", 21)]
        == "ray_hit_types[21](dx=1,dy=0,dz=0)"
    )


def test_feature_schema_rejects_invalid_group_shape():
    observation = radial_observation()
    schema = FeatureSchema.from_observation(observation)
    invalid = dict(observation)
    invalid["ray_hits"] = np.zeros(25, dtype=np.float32)

    with pytest.raises(ValueError, match="ray_hits"):
        schema.flatten(invalid)
