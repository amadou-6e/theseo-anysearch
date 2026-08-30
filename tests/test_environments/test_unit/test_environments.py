"""Unit tests for Gymnasium and PettingZoo environment wrappers and observation modes."""

import json
import re

import pytest
import numpy as np

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv
from theseo_anysearch.environments.pettingzoo.surface_env import SurfaceEnv


class TestVoxelEnv:
    """Tests VoxelEnv."""
    def make(self, **kwargs) -> VoxelEnv:
        cfg = {"agent_count": 2, "max_steps": 50, "seed": 42, **kwargs}
        return VoxelEnv(cfg)

    def test_observation_space_keys(self):
        env = self.make()
        assert set(env.observation_space.spaces) == set()

    def test_observation_space_shapes(self):
        env = self.make()
        assert len(env.observation_space.spaces) == 0

    def test_action_space_size(self):
        env = self.make()
        assert env.action_space.n == 26

    def test_reset_returns_obs_and_info(self):
        env = self.make()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "steps_remaining" not in obs
        assert isinstance(info, dict)

    def test_obs_values_are_numpy(self):
        env = self.make()
        obs, _ = env.reset()
        for v in obs.values():
            assert isinstance(v, np.ndarray)


class TestVoxelEnvObsModes:
    """Tests VoxelEnvObsModes."""
    def make(self, **kwargs) -> VoxelEnv:
        cfg = {"max_steps": 50, "seed": 42, **kwargs}
        return VoxelEnv(cfg)

    def make_waypoints_file(self, tmp_path):
        path = tmp_path.joinpath("waypoints.json")
        path.write_text(
            json.dumps({"start": [4, 4, 4], "goal": [4, 4, 6]}),
            encoding="utf-8",
        )
        return str(path)

    def test_scalar_mode_has_no_shortcut_fields(self):
        env = self.make(obs_mode="scalar")
        keys = set(env.observation_space.spaces)
        assert keys == set()

    def test_scalar_default_no_cursor_pos(self):
        env = self.make()  # no obs_mode key
        assert "cursor_pos" not in env.observation_space.spaces

    def test_box_obs_space_shape_radius2(self):
        env = self.make(obs_mode="box", box_radius=2)
        assert env.observation_space["local_grid"].shape == (125,)

    def test_box_obs_space_shape_radius3(self):
        env = self.make(obs_mode="box", box_radius=3)
        assert env.observation_space["local_grid"].shape == (343,)

    def test_box_cursor_pos_is_not_exposed(self):
        env = self.make(obs_mode="box")
        assert "cursor_pos" not in env.observation_space.spaces

    def test_goal_direction_in_keys_when_waypoints_present(self, tmp_path):
        env = self.make(obs_mode="box", waypoints_file=self.make_waypoints_file(tmp_path))
        assert "goal_direction" in env.observation_space.spaces
        assert env.observation_space["goal_direction"].shape == (3,)

    def test_radial_obs_space_shape(self):
        env = self.make(obs_mode="radial")
        assert env.observation_space["ray_hits"].shape == (26,)
        assert env.observation_space["ray_hit_types"].shape == (26,)

    def test_radial_cursor_pos_is_not_exposed(self):
        env = self.make(obs_mode="radial")
        assert "cursor_pos" not in env.observation_space.spaces

    def test_scalar_goal_keys_when_waypoints_present(self, tmp_path):
        env = self.make(obs_mode="scalar", waypoints_file=self.make_waypoints_file(tmp_path))
        assert "goal_distance" in env.observation_space.spaces
        assert "goal_direction" in env.observation_space.spaces

    def test_radial_no_local_grid_key(self):
        env = self.make(obs_mode="radial")
        assert "local_grid" not in env.observation_space.spaces

    def test_box_no_ray_hits_key(self):
        env = self.make(obs_mode="box")
        assert "ray_hits" not in env.observation_space.spaces

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            self.make(obs_mode="bogus")

    @pytest.mark.parametrize(
        "obs_mode", ["scalar", "box", "radial", "hierarchical_box"]
    )
    def test_shortcut_fields_are_absent_from_every_voxel_mode(self, obs_mode):
        env = self.make(obs_mode=obs_mode)

        assert "cursor_pos" not in env.observation_space.spaces
        assert "steps_remaining" not in env.observation_space.spaces


class TestMultiVoxelObservation:
    """Multi-agent voxel policies receive no absolute cursor position."""

    def test_cursor_pos_is_not_exposed(self):
        env = MultiVoxelEnv.__new__(MultiVoxelEnv)
        env._config = {"box_radius": 1}
        env._has_goal = lambda: False

        space = env._observation_space("agent_0")

        assert "cursor_pos" not in space.spaces


class TestVoxelEnvLoadWaypoints:
    """Tests VoxelEnv._load_waypoints failure modes (issue #118)."""

    def test_valid_waypoints_file_loads(self, tmp_path):
        path = tmp_path.joinpath("waypoints.json")
        path.write_text(
            json.dumps({"start": [4, 4, 4], "goal": [4, 4, 6]}),
            encoding="utf-8",
        )
        wp = VoxelEnv._load_waypoints(str(path))
        assert wp == {"start": [4, 4, 4], "goal": [4, 4, 6]}

    def test_missing_file_raises_with_path(self, tmp_path):
        missing = tmp_path.joinpath("does_not_exist.json")
        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            VoxelEnv._load_waypoints(str(missing))

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path.joinpath("waypoints.json")
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            VoxelEnv._load_waypoints(str(path))

    def test_missing_required_keys_raises(self, tmp_path):
        path = tmp_path.joinpath("waypoints.json")
        path.write_text(json.dumps({"start": [4, 4, 4]}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required key"):
            VoxelEnv._load_waypoints(str(path))

    def test_invalid_coordinate_shape_raises(self, tmp_path):
        path = tmp_path.joinpath("waypoints.json")
        path.write_text(
            json.dumps({"start": [4, 4], "goal": [4, 4, 6]}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="invalid 'start' value"):
            VoxelEnv._load_waypoints(str(path))

    def test_unconfigured_waypoints_file_does_not_raise(self):
        # waypoints_file left unset entirely: _build_rust_env must not call
        # _load_waypoints at all, and env construction must succeed.
        env = VoxelEnv(
            {"agent_count": 1, "max_steps": 50, "seed": 42, "waypoints_file": None}
        )
        assert "goal_distance" not in env.observation_space.spaces


class TestSurfaceEnv:
    """Tests SurfaceEnv."""
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
        for key in ("position", "target", "reached"):
            assert key in space.spaces

    def test_observation_space_shapes(self):
        env = self.make()
        space = env.observation_space("agent_0")
        assert "steps_remaining" not in space.spaces
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


def test_voxel_env_enforces_non_cubic_axis_bounds():
    from theseo_anysearch.environments.action_spaces import ACTION_OFFSETS_26

    env = VoxelEnv(
        {
            "grid_size": None,
            "extent": (7, 3, 2),
            "max_steps": 4,
            "trail_mode": False,
            "action_mode": "discrete_26",
            "waypoints": {"start": (7, 3, 2), "goal": (1, 1, 1)},
            "task": {},
        }
    )
    try:
        env.reset(seed=1)
        positive_y = ACTION_OFFSETS_26.index((0, 1, 0))
        _, _, _, _, info = env.step(positive_y)
        assert env._rust_env.cursor_pos() == (7, 3, 2)
        assert info["collision"] is True
    finally:
        env.close()
