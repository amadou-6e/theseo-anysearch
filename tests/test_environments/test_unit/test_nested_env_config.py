from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.models import EnvConfig, GeometryConfig, NestedFieldAccessMixin


def test_nested_blocks_resolve_to_runtime_environment() -> None:
    configured = EnvConfig(
        agent_count=1,
        max_steps=20,
        geometry={"grid_size": 8, "boxes": [[3, 3, 3, 3, 3, 3]]},
        observation={"mode": "box", "box_radius": 1},
        action={"mode": "discrete_26"},
        rewards={"step_cost": -0.02, "goal_reward": 2.0},
    )
    assert configured.rewards__step_cost == -0.02
    assert configured.geometry__boxes == [[3, 3, 3, 3, 3, 3]]
    assert configured.observation__mode == "box"
    assert configured.action__mode == "discrete_26"
    runtime = configured.to_runtime_dict()
    assert runtime["grid_size"] == 8
    assert runtime["geometry_boxes"] == [[3, 3, 3, 3, 3, 3]]
    assert runtime["obs_mode"] == "box"
    assert runtime["box_radius"] == 1
    assert runtime["action_mode"] == "discrete_26"
    assert runtime["step_cost"] == -0.02
    assert runtime["goal_reward"] == 2.0

    env = VoxelEnv(runtime)
    observation, _ = env.reset(seed=42)
    assert env.action_space.n == 26
    assert observation["local_grid"].shape == (27,)
    env.close()


def test_legacy_flattened_environment_remains_loadable() -> None:
    configured = EnvConfig(grid_size=8, obs_mode="radial", goal_reward=2.0)
    assert configured.geometry.grid_size == 8
    assert configured.observation.mode == "radial"
    assert configured.rewards.goal_reward == 2.0


def test_non_cubic_extent_is_preserved_for_regional_world_pipeline() -> None:
    configured = EnvConfig(geometry={"extent": [100, 50, 10]})

    assert configured.geometry.grid_size is None
    assert configured.geometry.extent == (100, 50, 10)
    assert configured.to_runtime_dict()["extent"] == (100, 50, 10)


def test_conflicting_grid_size_and_extent_are_rejected() -> None:
    with pytest.raises(ValidationError, match="describe different bounds"):
        EnvConfig(geometry={"grid_size": 32, "extent": [32, 64, 32]})


def test_mixed_legacy_and_nested_geometry_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be mixed.*geometry"):
        EnvConfig(grid_size=8, geometry={"grid_size": 16})


def test_mixed_legacy_and_nested_rewards_are_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be mixed.*rewards"):
        EnvConfig(goal_reward=2.0, rewards={"goal_reward": 3.0})


def test_compiled_world_navigation_requires_an_episode_source() -> None:
    with pytest.raises(ValueError, match="compiled-world navigation requires"):
        VoxelEnv(
            {
                "compiled_world_path": "unused",
                "extent": (64, 48, 32),
                "grid_size": None,
            }
        )

def test_nested_field_access_mixin_is_reusable() -> None:
    class GeometryWrapper(NestedFieldAccessMixin, BaseModel):
        exposed_nested_fields: ClassVar[tuple[str, ...]] = ("geometry",)
        geometry: GeometryConfig

    wrapped = GeometryWrapper(geometry=GeometryConfig(grid_size=64))
    assert wrapped.geometry__grid_size == 64
    assert "geometry__grid_size" not in wrapped.model_dump()
    with pytest.raises(AttributeError):
        _ = wrapped.grid_size
    with pytest.raises(AttributeError):
        _ = wrapped.unknown_attribute
