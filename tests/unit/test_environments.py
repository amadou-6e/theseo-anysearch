import pytest
import numpy as np

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.environments.pettingzoo.surface_env import SurfaceEnv


class TestVoxelEnv:
    def make(self, **kwargs) -> VoxelEnv:
        cfg = {"agent_count": 2, "max_steps": 50, "seed": 42, **kwargs}
        return VoxelEnv(cfg)

    def test_observation_space_keys(self):
        env = self.make()
        assert "steps_remaining" in env.observation_space.spaces
        assert "voxel_count" in env.observation_space.spaces

    def test_observation_space_shapes(self):
        env = self.make()
        assert env.observation_space["steps_remaining"].shape == (1,)
        assert env.observation_space["voxel_count"].shape == (1,)

    def test_action_space_size(self):
        env = self.make()
        assert env.action_space.n == 26

    def test_reset_returns_obs_and_info(self):
        env = self.make()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "steps_remaining" in obs
        assert "voxel_count" in obs
        assert isinstance(info, dict)

    def test_obs_values_are_numpy(self):
        env = self.make()
        obs, _ = env.reset()
        for v in obs.values():
            assert isinstance(v, np.ndarray)


class TestVoxelEnvObsModes:
    def make(self, **kwargs) -> VoxelEnv:
        cfg = {"max_steps": 50, "seed": 42, **kwargs}
        return VoxelEnv(cfg)

    def test_scalar_mode_unchanged(self):
        env = self.make(obs_mode="scalar")
        keys = set(env.observation_space.spaces)
        assert keys == {"steps_remaining", "voxel_count"}

    def test_scalar_default_no_cursor_pos(self):
        env = self.make()  # no obs_mode key
        assert "cursor_pos" not in env.observation_space.spaces

    def test_box_obs_space_shape_radius2(self):
        env = self.make(obs_mode="box", box_radius=2)
        assert env.observation_space["local_grid"].shape == (125,)

    def test_box_obs_space_shape_radius3(self):
        env = self.make(obs_mode="box", box_radius=3)
        assert env.observation_space["local_grid"].shape == (343,)

    def test_box_cursor_pos_in_keys(self):
        env = self.make(obs_mode="box")
        assert "cursor_pos" in env.observation_space.spaces
        assert env.observation_space["cursor_pos"].shape == (3,)

    def test_radial_obs_space_shape(self):
        env = self.make(obs_mode="radial")
        assert env.observation_space["ray_hits"].shape == (27,)

    def test_radial_cursor_pos_in_keys(self):
        env = self.make(obs_mode="radial")
        assert "cursor_pos" in env.observation_space.spaces
        assert env.observation_space["cursor_pos"].shape == (3,)

    def test_radial_no_local_grid_key(self):
        env = self.make(obs_mode="radial")
        assert "local_grid" not in env.observation_space.spaces

    def test_box_no_ray_hits_key(self):
        env = self.make(obs_mode="box")
        assert "ray_hits" not in env.observation_space.spaces

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            self.make(obs_mode="bogus")


class TestSurfaceEnv:
    def make(self, agent_count: int = 3) -> SurfaceEnv:
        cfg = {"agent_count": agent_count, "max_steps": 50, "seed": 42}
        return SurfaceEnv(cfg)

    def test_agent_ids_match_count(self):
        env = self.make(agent_count=4)
        assert env.possible_agents == ["agent_0", "agent_1", "agent_2", "agent_3"]

    def test_agents_default_to_possible(self):
        env = self.make(agent_count=2)
        assert env.agents == env.possible_agents

    def test_observation_space_keys(self):
        env = self.make()
        space = env.observation_space("agent_0")
        for key in ("steps_remaining", "position", "target", "reached"):
            assert key in space.spaces

    def test_observation_space_shapes(self):
        env = self.make()
        space = env.observation_space("agent_0")
        assert space["steps_remaining"].shape == (1,)
        assert space["position"].shape == (3,)
        assert space["target"].shape == (3,)

    def test_action_space_single(self):
        env = self.make()
        assert env.action_space("agent_0").n == 1

    def test_reset_returns_per_agent_obs(self):
        env = self.make(agent_count=2)
        obs, infos = env.reset()
        assert set(obs.keys()) == {"agent_0", "agent_1"}
        assert set(infos.keys()) == {"agent_0", "agent_1"}

    def test_reset_obs_are_dicts(self):
        env = self.make(agent_count=2)
        obs, _ = env.reset()
        for agent_obs in obs.values():
            assert isinstance(agent_obs, dict)
