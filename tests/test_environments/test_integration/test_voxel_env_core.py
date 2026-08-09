"""Integration tests for core single-agent voxel environment behavior."""

from __future__ import annotations

import numpy as np
import pytest

theseo_core = pytest.importorskip("theseo_core", reason="theseo_core wheel not installed")


@pytest.mark.integration
class TestVoxelEnvIntegration:
    """Verify core reset, step, and direct Rust behavior."""

    def test_rust_env_is_not_none(self, env):
        assert env._rust_env is not None

    def test_rust_env_is_py_voxel_env(self, env):
        assert type(env._rust_env).__name__ == "PyVoxelEnv"

    def test_observation_space_has_expected_keys(self, env):
        assert set(env.observation_space.spaces) == set()

    def test_reset_returns_dict_with_correct_keys(self, env):
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == set()
        assert isinstance(info, dict)

    def test_reset_obs_dtypes_are_float32(self, env):
        obs, _ = env.reset(seed=0)
        assert "steps_remaining" not in obs

    def test_reset_steps_remaining_is_not_exposed(self, env):
        obs, _ = env.reset(seed=0)
        assert "steps_remaining" not in obs

    def test_step_returns_five_tuple(self, env):
        env.reset(seed=0)
        assert len(env.step(0)) == 5

    def test_step_reward_is_float(self, env):
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_step_noop_reward_is_below_zero_point_three(self, env):
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(2)
        assert reward < 0.3

    def test_episode_terminates_within_max_steps(self, env):
        env.reset(seed=0)
        done = False
        for _ in range(21):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                done = True
                break
        assert done

    def test_encode_action_returns_int(self, env):
        assert env._encode_action(0) == 0
        assert env._encode_action(np.int64(2)) == 2

    def test_direct_rust_env_reset_observation(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=10)
        obs = rust_env.reset(seed=99)
        assert obs.filled == 0
        assert obs.steps_remaining == 10

    def test_direct_rust_env_step_decrements_steps_remaining(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=10)
        rust_env.reset(seed=0)
        result = rust_env.step(0)
        assert result.observation.steps_remaining == 9
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_direct_rust_env_done_at_max_steps(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=3)
        rust_env.reset(seed=0)
        rust_env.step(2)
        rust_env.step(2)
        assert rust_env.step(2).done is True

    def test_direct_rust_env_move_increases_fill(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=20)
        obs0 = rust_env.reset(seed=0)
        result = rust_env.step(21)
        assert result.observation.filled == obs0.filled + 1

    def test_direct_rust_env_filled_cell_blocks_move(self):
        rust_env = theseo_core.PyVoxelEnv(max_steps=40)
        rust_env.reset(seed=0)
        result1 = rust_env.step(21)
        result2 = rust_env.step(4)
        result3 = rust_env.step(21)
        assert result1.observation.filled == 1
        assert result2.observation.filled == 2
        assert result3.observation.filled == 2
