"""Built-in voxel perception implementations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from gymnasium import spaces
from pydantic import Field, field_validator

from theseo_anysearch.environments.perception.base import (
    Perception,
    PerceptionMetadata,
    PerceptionParameters,
)
from theseo_anysearch.environments.perception.registry import register_perception


MAX_RAY_HIT_TYPE = 5.0


class LocalBoxParameters(PerceptionParameters):
    radius: int = Field(default=2, ge=0)


class RadialParameters(PerceptionParameters):
    max_len: int = Field(default=16, ge=1)


class HierarchicalBoxParameters(PerceptionParameters):
    radii: list[int] = Field(default_factory=lambda: [1, 4])

    @field_validator("radii")
    @classmethod
    def validate_radii(cls, value: list[int]) -> list[int]:
        if not value or any(radius < 0 for radius in value):
            raise ValueError("radii must be a non-empty list of non-negative integers")
        return value


class GoalVectorPerception(Perception):
    metadata = PerceptionMetadata(
        name="goal_vector",
        normalization="distance divided by maximum grid Manhattan distance; direction in [-1, 1]",
        required_capabilities=("goal",),
    )

    def observation_space(self, env: Any) -> Mapping[str, spaces.Space]:
        if not env._has_goal():
            return {}
        return {
            "goal_distance": spaces.Box(0.0, 1.0, (1,), np.float32),
            "goal_direction": spaces.Box(-1.0, 1.0, (3,), np.float32),
        }

    def compute(self, env: Any, native_observation: Any) -> Mapping[str, Any]:
        if not env._has_goal_flag or native_observation.goal_distance is None:
            return {}
        distance = np.asarray(
            [native_observation.goal_distance * env._inv_max_manhattan], dtype=np.float32
        )
        values: dict[str, Any] = {"goal_distance": distance}
        if native_observation.goal_direction is not None:
            values["goal_direction"] = np.asarray(
                native_observation.goal_direction, dtype=np.float32
            )
        return values


class LocalBoxPerception(Perception):
    parameter_model = LocalBoxParameters
    metadata = PerceptionMetadata(
        name="local_box",
        normalization="binary occupancy in [0, 1]",
        required_capabilities=("voxel_grid", "cursor"),
        compatible_models=("voxel_encoder", "voxel_box_2d_cnn", "voxel_box_3d_cnn"),
        estimated_compute="O((2r+1)^3)",
    )

    def _radius(self) -> int:
        return self.config.radius

    def observation_space(self, env: Any) -> Mapping[str, spaces.Space]:
        size = (2 * self._radius() + 1) ** 3
        return {"local_grid": spaces.Box(0.0, 1.0, (size,), np.float32)}

    def compute(self, env: Any, native_observation: Any) -> Mapping[str, Any]:
        radius = self._radius()
        values = native_observation.local_grid
        if values is None or radius != getattr(env, "_native_box_radius", None):
            values = env._rust_env.box_obs(radius)
        return {"local_grid": (np.asarray(values) > 0).astype(np.float32)}


class RadialPerception(Perception):
    parameter_model = RadialParameters
    metadata = PerceptionMetadata(
        name="radial",
        normalization="hit proximity and hit type in [0, 1]",
        required_capabilities=("voxel_grid", "cursor"),
        estimated_compute="O(26 * max_len)",
    )

    def _max_len(self) -> int:
        return self.config.max_len

    def observation_space(self, env: Any) -> Mapping[str, spaces.Space]:
        return {
            "ray_hits": spaces.Box(0.0, 1.0, (26,), np.float32),
            "ray_hit_types": spaces.Box(0.0, 1.0, (26,), np.float32),
        }

    def compute(self, env: Any, native_observation: Any) -> Mapping[str, Any]:
        max_len = self._max_len()
        return {
            "ray_hits": np.asarray(env._rust_env.radial_obs(max_len), dtype=np.float32),
            "ray_hit_types": np.asarray(
                env._rust_env.radial_obs_types(max_len), dtype=np.float32
            ) / MAX_RAY_HIT_TYPE,
        }


class HierarchicalBoxPerception(Perception):
    parameter_model = HierarchicalBoxParameters
    metadata = PerceptionMetadata(
        name="hierarchical_box",
        normalization="concatenated binary occupancy in [0, 1]",
        required_capabilities=("voxel_grid", "cursor"),
        compatible_models=("voxel_encoder", "voxel_hierarchical_box_2d_cnn", "voxel_hierarchical_box_3d_cnn"),
        estimated_compute="O(sum((2r+1)^3))",
    )

    def _radii(self) -> tuple[int, ...]:
        return tuple(self.config.radii)

    def observation_space(self, env: Any) -> Mapping[str, spaces.Space]:
        size = sum((2 * radius + 1) ** 3 for radius in self._radii())
        return {"local_grid": spaces.Box(0.0, 1.0, (size,), np.float32)}

    def compute(self, env: Any, native_observation: Any) -> Mapping[str, Any]:
        parts = [
            (np.asarray(env._rust_env.box_obs(radius)) > 0).astype(np.float32)
            for radius in self._radii()
        ]
        return {"local_grid": np.concatenate(parts)}


register_perception("goal_vector", GoalVectorPerception)
register_perception("local_box", LocalBoxPerception)
register_perception("radial", RadialPerception)
register_perception("hierarchical_box", HierarchicalBoxPerception)
