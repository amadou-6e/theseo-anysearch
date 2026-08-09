"""Contract tests for named and composable perceptions."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from theseo_anysearch.environments.perception import (
    Perception,
    PerceptionMetadata,
    PerceptionParameters,
    PerceptionPipeline,
    get_perception,
    perception_names,
    register_perception,
)
from theseo_anysearch.settings import EnvConfig, ObservationConfig


def test_builtins_publish_stable_metadata():
    assert {"goal_vector", "local_box", "radial", "hierarchical_box"} <= set(
        perception_names()
    )
    metadata = get_perception("radial").metadata
    assert metadata.schema_version == "1"
    assert metadata.dtype == "float32"
    assert metadata.required_capabilities == ("voxel_grid", "cursor")


def test_unknown_sensor_fails_during_settings_validation():
    with pytest.raises(ValueError, match="unknown perception"):
        ObservationConfig(sensors=[{"name": "missing"}])


def test_invalid_sensor_parameters_fail_during_settings_validation():
    with pytest.raises(ValueError, match="radius"):
        ObservationConfig(
            sensors=[{"name": "local_box", "parameters": {"radius": -1}}]
        )


def test_duplicate_sensors_fail_during_settings_validation():
    with pytest.raises(ValueError, match="duplicate perceptions"):
        ObservationConfig(sensors=[{"name": "radial"}, {"name": "radial"}])


def test_explicit_goal_sensor_requires_goal_capability():
    with pytest.raises(ValueError, match="requires a goal-capable environment"):
        EnvConfig(
            agent_count=1,
            observation={"sensors": [{"name": "goal_vector"}]},
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("scalar", ["goal_vector"]),
        ("box", ["local_box", "goal_vector"]),
        ("radial", ["radial", "goal_vector"]),
        ("hierarchical_box", ["hierarchical_box", "goal_vector"]),
    ],
)
def test_legacy_modes_resolve_to_equivalent_sensor_lists(mode, expected):
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = ObservationConfig(mode=mode)
    assert [selector.name for selector in config.resolved_selectors()] == expected


def test_new_and_legacy_selection_cannot_be_mixed():
    with pytest.raises(ValueError, match="cannot be combined"):
        ObservationConfig(mode="radial", sensors=[{"name": "radial"}])


def test_third_party_sensor_registration_and_composition():
    class ConstantParameters(PerceptionParameters):
        value: float

    class ConstantPerception(Perception):
        metadata = PerceptionMetadata(name="test_constant")
        parameter_model = ConstantParameters

        def observation_space(self, env):
            return {"constant": spaces.Box(0.0, 1.0, (1,), np.float32)}

        def compute(self, env, native_observation):
            return {"constant": np.asarray([self.parameters["value"]], dtype=np.float32)}

    if "test_constant" not in perception_names():
        register_perception("test_constant", ConstantPerception)
    pipeline = PerceptionPipeline(
        [{"name": "test_constant", "parameters": {"value": 0.25}}]
    )
    assert pipeline.observation_space(object())["constant"].shape == (1,)
    assert pipeline.compute(object(), object())["constant"].tolist() == [0.25]
    assert pipeline.metadata()[0]["parameters"] == {"value": 0.25}


def test_feature_key_collisions_fail_before_training():
    class DuplicateFeaturePerception(Perception):
        metadata = PerceptionMetadata(name="test_duplicate_feature")

        def observation_space(self, env):
            return {"ray_hits": spaces.Box(0.0, 1.0, (26,), np.float32)}

        def compute(self, env, native_observation):
            return {}

    if "test_duplicate_feature" not in perception_names():
        register_perception("test_duplicate_feature", DuplicateFeaturePerception)
    with pytest.raises(ValueError, match="duplicate feature"):
        ObservationConfig(
            sensors=[{"name": "radial"}, {"name": "test_duplicate_feature"}]
        )
