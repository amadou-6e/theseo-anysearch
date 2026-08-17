"""Integration tests for voxel environment observation modes."""

from __future__ import annotations

import numpy as np
import pytest

theseo_core = pytest.importorskip("theseo_core", reason="theseo_core wheel not installed")

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv


@pytest.mark.integration
class TestVoxelEnvObsModesIntegration:
    """Verify box, radial, and cursor observation behavior."""

    def make(self, **kwargs) -> VoxelEnv:
        cfg = {"max_steps": 20, "seed": 42, **kwargs}
        return VoxelEnv(cfg)

    def test_box_obs_shape_after_reset(self):
        env = self.make(obs_mode="box", box_radius=2)
        obs, _ = env.reset()
        assert obs["local_grid"].shape == (125,)

    def test_box_obs_shape_after_step(self):
        env = self.make(obs_mode="box", box_radius=2)
        env.reset()
        obs, *_ = env.step(2)
        assert obs["local_grid"].shape == (125,)

    def test_box_obs_values_are_normalized_block_kinds(self):
        env = self.make(obs_mode="box", box_radius=2)
        obs, _ = env.reset()
        valid = np.arange(6, dtype=np.float32) / 5.0
        assert np.all(np.isin(obs["local_grid"], valid))
        assert 0.8 in obs["local_grid"]  # outside-grid boundary

    def test_box_obs_dtype(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        assert obs["local_grid"].dtype == np.float32

    def test_box_move_fills_destination_visible_in_local_grid(self):
        env = self.make(obs_mode="box", box_radius=2)
        env.reset()
        obs, *_ = env.step(21)
        assert obs["local_grid"][62] == 1.0

    def test_radial_obs_shape(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        assert obs["ray_hits"].shape == (26,)

    def test_radial_obs_values_in_range(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        hits = obs["ray_hits"]
        assert np.all((hits >= 0.0) & (hits <= 1.0))

    def test_radial_obs_dtype(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        assert obs["ray_hits"].dtype == np.float32

    def test_radial_empty_world_shows_grid_boundary(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        assert np.any(obs["ray_hits"] > 0.0)

    def test_radial_shape_after_step(self):
        env = self.make(obs_mode="radial")
        env.reset()
        obs, *_ = env.step(2)
        assert obs["ray_hits"].shape == (26,)

    @pytest.mark.parametrize(
        "obs_mode", ["scalar", "box", "radial", "hierarchical_box"]
    )
    def test_filled_cell_count_is_not_exposed_to_policy(self, obs_mode):
        env = self.make(obs_mode=obs_mode)

        obs, _ = env.reset()

        assert "voxel_count" not in obs
        assert "voxel_count" not in env.observation_space.spaces

    def test_cursor_pos_not_in_box_obs(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        assert "cursor_pos" not in obs

    def test_cursor_pos_not_in_radial_obs(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        assert "cursor_pos" not in obs

    def test_steps_remaining_not_in_box_obs(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        assert "steps_remaining" not in obs

    def test_steps_remaining_not_in_radial_obs(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        assert "steps_remaining" not in obs

    def test_native_cursor_still_changes_after_step(self):
        env = self.make(obs_mode="box")
        obs0, _ = env.reset()
        obs1, *_ = env.step(21)
        assert env._rust_env.cursor_pos() != (1, 1, 1)

    def test_direct_cursor_pos_after_reset(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        rust_env.reset(seed=0)
        assert rust_env.cursor_pos() == (1, 1, 1)

    def test_direct_box_obs_empty_world(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        rust_env.reset(seed=0)
        obs = np.array(rust_env.box_obs(2), dtype=np.float32)
        expected = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    x, y, z = 1 + dx, 1 + dy, 1 + dz
                    expected.append(4.0 if x < 1 or y < 1 or z < 1 else 0.0)
        expected = np.array(expected, dtype=np.float32)
        expected[62] = 2.0
        assert len(obs) == 125
        assert np.array_equal(obs, expected)

    def test_direct_radial_obs_empty_world_shows_grid_boundary(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        rust_env.reset(seed=0)
        obs = np.array(rust_env.radial_obs(16), dtype=np.float32)
        assert len(obs) == 26
        assert obs[4] == pytest.approx(1.0)
