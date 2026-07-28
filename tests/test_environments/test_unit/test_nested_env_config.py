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


def test_mixed_legacy_and_nested_geometry_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be mixed.*geometry"):
        EnvConfig(grid_size=8, geometry={"grid_size": 16})


def test_mixed_legacy_and_nested_rewards_are_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be mixed.*rewards"):
        EnvConfig(goal_reward=2.0, rewards={"goal_reward": 3.0})

def test_nested_field_access_mixin_is_reusable() -> None:
    class GeometryWrapper(NestedFieldAccessMixin, BaseModel):
        exposed_nested_fields: ClassVar[dict[str, tuple[str, ...]]] = {
            "geometry": ("grid_size",),
        }
        exposed_nested_aliases: ClassVar[dict[str, tuple[str, str]]] = {
            "resolution": ("geometry", "grid_size"),
        }
        geometry: GeometryConfig

    wrapped = GeometryWrapper(geometry=GeometryConfig(grid_size=64))
    assert wrapped.resolution == 64
    assert "resolution" not in wrapped.model_dump()
    with pytest.raises(AttributeError):
        _ = wrapped.unknown_attribute