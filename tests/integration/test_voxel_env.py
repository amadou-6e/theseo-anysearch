"""
Integration tests for VoxelEnv — require theseo_core wheel.
Run with: pytest tests/integration/ -m integration
"""
import numpy as np
import pytest

theseo_core = pytest.importorskip("theseo_core", reason="theseo_core wheel not installed")

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv  # noqa: E402


VOXEL_CONFIG = {
    "max_steps": 20,
    "seed": 42,
}


@pytest.fixture()
def env():
    return VoxelEnv(VOXEL_CONFIG)


@pytest.mark.integration
class TestVoxelEnvIntegration:
    def test_rust_env_is_not_none(self, env):
        """_build_rust_env must return a live theseo_core.PyVoxelEnv instance."""
        assert env._rust_env is not None

    def test_rust_env_is_py_voxel_env(self, env):
        assert type(env._rust_env).__name__ == "PyVoxelEnv"

    def test_observation_space_has_expected_keys(self, env):
        assert "steps_remaining" in env.observation_space.spaces
        assert "voxel_count" in env.observation_space.spaces

    def test_reset_returns_dict_with_correct_keys(self, env):
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == {"steps_remaining", "voxel_count"}
        assert isinstance(info, dict)

    def test_reset_obs_dtypes_are_float32(self, env):
        obs, _ = env.reset(seed=0)
        assert obs["steps_remaining"].dtype == np.float32
        assert obs["voxel_count"].dtype == np.float32

    def test_reset_steps_remaining_is_normalised(self, env):
        obs, _ = env.reset(seed=0)
        # Normalised to [0, 1]: should be exactly 1.0 right after reset.
        assert 0.0 <= float(obs["steps_remaining"][0]) <= 1.0

    def test_step_returns_five_tuple(self, env):
        env.reset(seed=0)
        result = env.step(0)  # Place
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_step_reward_is_float(self, env):
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_step_noop_reward_is_below_zero_point_three(self, env):
        """Noop gives -0.01 base reward (plus shaping), so result < 0.3."""
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(2)  # Noop
        assert reward < 0.3

    def test_episode_terminates_within_max_steps(self, env):
        env.reset(seed=0)
        done = False
        for _ in range(VOXEL_CONFIG["max_steps"] + 1):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                done = True
                break
        assert done

    def test_voxel_count_nonnegative_after_reset(self, env):
        obs, _ = env.reset(seed=0)
        assert float(obs["voxel_count"][0]) >= 0.0

    def test_encode_action_returns_int(self, env):
        assert env._encode_action(0) == 0
        assert env._encode_action(np.int64(2)) == 2

    def test_direct_rust_env_reset_observation(self):
        """Call the Rust PyVoxelEnv directly — empty world has filled=0."""
        rust_env = theseo_core.PyVoxelEnv(max_steps=10)
        obs = rust_env.reset(seed=99)
        assert obs.filled == 0
        assert obs.steps_remaining == 10

    def test_direct_rust_env_step_decrements_steps_remaining(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=10)
        rust_env.reset(seed=0)
        result = rust_env.step(0)  # Place
        assert result.observation.steps_remaining == 9
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_direct_rust_env_done_at_max_steps(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=3)
        rust_env.reset(seed=0)
        rust_env.step(2)  # Noop
        rust_env.step(2)  # Noop
        result = rust_env.step(2)  # Noop — step 3 of 3
        assert result.done is True

    def test_direct_rust_env_move_increases_fill(self):
        """Trail mode: moving to an empty cell auto-fills the destination."""
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        obs0 = rust_env.reset(seed=0)
        result = rust_env.step(21)  # action 21 = (+1,0,0): move from (1,1,1) to (2,1,1)
        assert result.observation.filled == obs0.filled + 1

    def test_direct_rust_env_filled_cell_blocks_move(self):
        """Moving into an already-filled cell is a collision — fill count unchanged."""
        rust_env = theseo_core.PyVoxelEnv(max_steps=40)
        rust_env.reset(seed=0)
        r1 = rust_env.step(21)  # (+1,0,0): cursor (1,1,1)→(2,1,1), fills (2,1,1), fill=1
        r2 = rust_env.step(4)   # (-1,0,0): cursor (2,1,1)→(1,1,1), fills (1,1,1), fill=2
        r3 = rust_env.step(21)  # (+1,0,0): cursor tries (2,1,1) — already filled → collision
        assert r1.observation.filled == 1
        assert r2.observation.filled == 2
        assert r3.observation.filled == 2  # collision: count unchanged


@pytest.mark.integration
class TestVoxelEnvObsModesIntegration:
    def make(self, **kwargs) -> VoxelEnv:
        cfg = {"max_steps": 20, "seed": 42, **kwargs}
        return VoxelEnv(cfg)

    # --- box mode ---

    def test_box_obs_shape_after_reset(self):
        env = self.make(obs_mode="box", box_radius=2)
        obs, _ = env.reset()
        assert obs["local_grid"].shape == (125,)

    def test_box_obs_shape_after_step(self):
        env = self.make(obs_mode="box", box_radius=2)
        env.reset()
        obs, *_ = env.step(2)  # Noop
        assert obs["local_grid"].shape == (125,)

    def test_box_obs_values_binary(self):
        env = self.make(obs_mode="box", box_radius=2)
        obs, _ = env.reset()
        grid = obs["local_grid"]
        assert np.all((grid == 0.0) | (grid == 1.0))

    def test_box_obs_dtype(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        assert obs["local_grid"].dtype == np.float32

    def test_box_move_fills_destination_visible_in_local_grid(self):
        """Trail mode: after moving, the destination cell is filled and visible at centre."""
        env = self.make(obs_mode="box", box_radius=2)
        env.reset()
        obs, *_ = env.step(21)  # (+1,0,0): cursor moves to (2,1,1), fills it
        # Cursor is now at (2,1,1); centre of box_obs = index 2*25+2*5+2=62 should be 1.0
        assert obs["local_grid"][62] == 1.0

    # --- radial mode ---

    def test_radial_obs_shape(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        assert obs["ray_hits"].shape == (27,)

    def test_radial_obs_values_in_range(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        hits = obs["ray_hits"]
        assert np.all((hits >= 0.0) & (hits <= 1.0))

    def test_radial_obs_dtype(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        assert obs["ray_hits"].dtype == np.float32

    def test_radial_empty_world_all_zeros(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()  # fresh VoxelEnv starts empty
        assert np.all(obs["ray_hits"] == 0.0)

    def test_radial_self_cell_filled_index_13(self):
        """Trail mode: after moving, the destination cell is filled; direction (0,0,0) = index 13."""
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        rust_env.reset(seed=0)
        assert rust_env.cursor_pos() == (1, 1, 1)
        rust_env.step(21)  # (+1,0,0): cursor moves to (2,1,1), fills (2,1,1)
        # Cursor is now at (2,1,1); (0,0,0) direction = index 13: cursor cell is filled
        hits = np.array(rust_env.radial_obs(16))
        assert hits[13] == 1.0

    def test_radial_shape_after_step(self):
        env = self.make(obs_mode="radial")
        env.reset()
        obs, *_ = env.step(2)
        assert obs["ray_hits"].shape == (27,)

    # --- cursor_pos ---

    def test_cursor_pos_in_box_obs(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        assert "cursor_pos" in obs
        assert obs["cursor_pos"].shape == (3,)

    def test_cursor_pos_in_radial_obs(self):
        env = self.make(obs_mode="radial")
        obs, _ = env.reset()
        assert "cursor_pos" in obs

    def test_cursor_pos_normalised(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        pos = obs["cursor_pos"]
        assert np.all((pos >= 0.0) & (pos <= 1.0))

    def test_cursor_pos_dtype(self):
        env = self.make(obs_mode="box")
        obs, _ = env.reset()
        assert obs["cursor_pos"].dtype == np.float32

    def test_cursor_pos_changes_after_step(self):
        env = self.make(obs_mode="box")
        obs0, _ = env.reset()
        obs1, *_ = env.step(2)  # Move +x — cursor x changes
        assert not np.array_equal(obs0["cursor_pos"], obs1["cursor_pos"])

    # --- direct Rust cursor_pos / box_obs / radial_obs ---

    def test_direct_cursor_pos_after_reset(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        rust_env.reset(seed=0)
        pos = rust_env.cursor_pos()
        assert pos == (1, 1, 1)

    def test_direct_box_obs_empty_world(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        rust_env.reset(seed=0)
        obs = rust_env.box_obs(2)
        assert len(obs) == 125
        assert all(v == 0.0 for v in obs)

    def test_direct_radial_obs_empty_world(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        rust_env.reset(seed=0)
        obs = rust_env.radial_obs(16)
        assert len(obs) == 27
        assert all(v == 0.0 for v in obs)
