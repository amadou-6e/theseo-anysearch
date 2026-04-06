"""
Unit tests for VoxelBox2DCNN and VoxelBox3DCNN.

These tests instantiate the model classes directly without going through RLlib
registration or Ray. No Rust wheel required.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs_space(box_radius: int) -> gym.spaces.Dict:
    n = 2 * box_radius + 1
    return gym.spaces.Dict({
        "steps_remaining": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "voxel_count":     gym.spaces.Box(0.0, np.inf, (1,), np.float32),
        "cursor_pos":      gym.spaces.Box(0.0, 1.0, (3,), np.float32),
        "local_grid":      gym.spaces.Box(0.0, 1.0, (n**3,), np.float32),
    })


def _make_action_space() -> gym.spaces.Discrete:
    return gym.spaces.Discrete(3)


def _make_dummy_obs(batch: int, box_radius: int) -> dict[str, torch.Tensor]:
    n = 2 * box_radius + 1
    return {
        "steps_remaining": torch.rand(batch, 1),
        "voxel_count":     torch.rand(batch, 1),
        "cursor_pos":      torch.rand(batch, 3),
        "local_grid":      torch.rand(batch, n**3),
    }


def _build_2d(box_radius: int = 2, num_outputs: int = 3, **cfg_overrides):
    from theseo_anysearch.rllib.models.cnn import VoxelBox2DCNN

    obs_space = _make_obs_space(box_radius)
    act_space = _make_action_space()
    model_config = {
        "custom_model_config": {
            "box_radius": box_radius,
            **cfg_overrides,
        }
    }
    return VoxelBox2DCNN(obs_space, act_space, num_outputs, model_config, "test_2d")


def _build_3d(box_radius: int = 2, num_outputs: int = 3, **cfg_overrides):
    from theseo_anysearch.rllib.models.cnn import VoxelBox3DCNN

    obs_space = _make_obs_space(box_radius)
    act_space = _make_action_space()
    model_config = {
        "custom_model_config": {
            "box_radius": box_radius,
            **cfg_overrides,
        }
    }
    return VoxelBox3DCNN(obs_space, act_space, num_outputs, model_config, "test_3d")


def _forward(model, box_radius: int = 2, batch: int = 1):
    obs = _make_dummy_obs(batch, box_radius)
    input_dict = {"obs": obs}
    logits, state = model.forward(input_dict, [], torch.tensor([batch]))
    value = model.value_function()
    return logits, value, state


# ---------------------------------------------------------------------------
# 2D CNN tests
# ---------------------------------------------------------------------------

class TestVoxelBox2DCNN:
    def test_forward_output_shape_radius2(self):
        model = _build_2d(box_radius=2, num_outputs=3)
        logits, _, _ = _forward(model, box_radius=2, batch=1)
        assert logits.shape == (1, 3)

    def test_forward_output_shape_radius3(self):
        model = _build_2d(box_radius=3, num_outputs=3)
        logits, _, _ = _forward(model, box_radius=3, batch=1)
        assert logits.shape == (1, 3)

    def test_value_function_shape(self):
        model = _build_2d()
        _, value, _ = _forward(model)
        assert value.shape == (1,)

    def test_batch_size_2(self):
        model = _build_2d(num_outputs=3)
        logits, value, _ = _forward(model, batch=2)
        assert logits.shape == (2, 3)
        assert value.shape == (2,)

    def test_logits_are_finite(self):
        model = _build_2d()
        logits, _, _ = _forward(model)
        assert torch.isfinite(logits).all()

    def test_value_is_finite(self):
        model = _build_2d()
        _, value, _ = _forward(model)
        assert torch.isfinite(value).all()

    def test_state_is_empty_list(self):
        model = _build_2d()
        _, _, state = _forward(model)
        assert state == []

    def test_custom_channels(self):
        model = _build_2d(conv_channels=[16, 32])
        logits, _, _ = _forward(model)
        assert logits.shape == (1, 3)

    def test_two_fc_hiddens(self):
        model = _build_2d(fc_hiddens=[128, 64])
        logits, _, _ = _forward(model)
        assert logits.shape == (1, 3)

    def test_max_pool_variant(self):
        model = _build_2d(pool_type="max")
        logits, _, _ = _forward(model)
        assert logits.shape == (1, 3)

    def test_different_num_outputs(self):
        model = _build_2d(num_outputs=7)
        logits, _, _ = _forward(model)
        assert logits.shape == (1, 7)


# ---------------------------------------------------------------------------
# 3D CNN tests
# ---------------------------------------------------------------------------

class TestVoxelBox3DCNN:
    def test_forward_output_shape_radius2(self):
        model = _build_3d(box_radius=2, num_outputs=3)
        logits, _, _ = _forward(model, box_radius=2)
        assert logits.shape == (1, 3)

    def test_forward_output_shape_radius3(self):
        model = _build_3d(box_radius=3, num_outputs=3)
        logits, _, _ = _forward(model, box_radius=3)
        assert logits.shape == (1, 3)

    def test_value_function_shape(self):
        model = _build_3d()
        _, value, _ = _forward(model)
        assert value.shape == (1,)

    def test_batch_size_2(self):
        model = _build_3d(num_outputs=3)
        logits, value, _ = _forward(model, batch=2)
        assert logits.shape == (2, 3)
        assert value.shape == (2,)

    def test_logits_are_finite(self):
        model = _build_3d()
        logits, _, _ = _forward(model)
        assert torch.isfinite(logits).all()

    def test_value_is_finite(self):
        model = _build_3d()
        _, value, _ = _forward(model)
        assert torch.isfinite(value).all()

    def test_state_is_empty_list(self):
        model = _build_3d()
        _, _, state = _forward(model)
        assert state == []

    def test_custom_channels(self):
        model = _build_3d(conv_channels=[16, 32])
        logits, _, _ = _forward(model)
        assert logits.shape == (1, 3)

    def test_two_fc_hiddens(self):
        model = _build_3d(fc_hiddens=[128, 64])
        logits, _, _ = _forward(model)
        assert logits.shape == (1, 3)

    def test_max_pool_variant(self):
        model = _build_3d(pool_type="max")
        logits, _, _ = _forward(model)
        assert logits.shape == (1, 3)

    def test_3d_vs_2d_same_output_dim(self):
        m2 = _build_2d(num_outputs=4)
        m3 = _build_3d(num_outputs=4)
        l2, _, _ = _forward(m2)
        l3, _, _ = _forward(m3)
        assert l2.shape == l3.shape == (1, 4)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestModelRegistration:
    def test_register_function_runs(self):
        from theseo_anysearch.rllib.models.cnn import register_voxel_cnn_models
        register_voxel_cnn_models()  # should not raise

    def test_both_keys_registered(self):
        from ray.rllib.models.catalog import RLLIB_MODEL, _global_registry
        from theseo_anysearch.rllib.models.cnn import register_voxel_cnn_models
        register_voxel_cnn_models()
        assert _global_registry.contains(RLLIB_MODEL, "voxel_box_2d_cnn")
        assert _global_registry.contains(RLLIB_MODEL, "voxel_box_3d_cnn")


# ---------------------------------------------------------------------------
# build_rllib_model_dict helper tests
# ---------------------------------------------------------------------------

class TestBuildRllibModelDict:
    def setup_method(self):
        from theseo_anysearch.rllib.models import build_rllib_model_dict
        from theseo_anysearch.models import ModelConfig
        self._build = build_rllib_model_dict
        self.ModelConfig = ModelConfig

    def test_fcnet_path_when_no_custom_model(self):
        cfg = self.ModelConfig(hidden_sizes=[128, 64], activation="tanh")
        result = self._build(cfg)
        assert result == {"fcnet_hiddens": [128, 64], "fcnet_activation": "tanh"}

    def test_custom_model_path(self):
        cfg = self.ModelConfig(
            custom_model="voxel_box_3d_cnn",
            custom_model_config={"box_radius": 2},
        )
        result = self._build(cfg)
        assert result["custom_model"] == "voxel_box_3d_cnn"
        assert result["custom_model_config"] == {"box_radius": 2}

    def test_custom_model_config_defaults_to_empty_dict(self):
        cfg = self.ModelConfig(custom_model="voxel_box_2d_cnn")
        result = self._build(cfg)
        assert result["custom_model_config"] == {}


# ---------------------------------------------------------------------------
# Hierarchical box CNN helpers
# ---------------------------------------------------------------------------

def _make_hier_obs_space(radii: list[int]) -> gym.spaces.Dict:
    flat_size = sum((2 * r + 1) ** 3 for r in radii)
    return gym.spaces.Dict({
        "steps_remaining": gym.spaces.Box(0.0, 1.0,    (1,),         np.float32),
        "voxel_count":     gym.spaces.Box(0.0, np.inf,  (1,),         np.float32),
        "cursor_pos":      gym.spaces.Box(0.0, 1.0,    (3,),         np.float32),
        "local_grid":      gym.spaces.Box(0.0, 1.0,    (flat_size,), np.float32),
    })


def _make_hier_obs(batch: int, radii: list[int]) -> dict[str, torch.Tensor]:
    flat_size = sum((2 * r + 1) ** 3 for r in radii)
    return {
        "steps_remaining": torch.rand(batch, 1),
        "voxel_count":     torch.rand(batch, 1),
        "cursor_pos":      torch.rand(batch, 3),
        "local_grid":      torch.rand(batch, flat_size),
    }


def _build_hier(radii: list[int] = None, num_outputs: int = 3, **cfg_overrides):
    from theseo_anysearch.rllib.models.cnn import VoxelHierarchicalBox3DCNN

    radii = radii or [1, 4]
    obs_space = _make_hier_obs_space(radii)
    act_space = _make_action_space()
    model_config = {
        "custom_model_config": {
            "box_radii": radii,
            **cfg_overrides,
        }
    }
    return VoxelHierarchicalBox3DCNN(obs_space, act_space, num_outputs, model_config, "test_hier")


def _forward_hier(model, radii: list[int] = None, batch: int = 1):
    radii = radii or [1, 4]
    obs = _make_hier_obs(batch, radii)
    input_dict = {"obs": obs}
    logits, state = model.forward(input_dict, [], torch.tensor([batch]))
    value = model.value_function()
    return logits, value, state


# ---------------------------------------------------------------------------
# VoxelHierarchicalBox3DCNN tests
# ---------------------------------------------------------------------------

class TestVoxelHierarchicalBox3DCNN:
    def test_output_shape_default_radii(self):
        m = _build_hier()
        logits, _, _ = _forward_hier(m)
        assert logits.shape == (1, 3)

    def test_output_shape_three_radii(self):
        m = _build_hier(radii=[1, 2, 4])
        logits, _, _ = _forward_hier(m, radii=[1, 2, 4])
        assert logits.shape == (1, 3)

    def test_value_function_shape_batch1(self):
        m = _build_hier()
        _, value, _ = _forward_hier(m)
        assert value.shape == (1,)

    def test_value_function_shape_batch4(self):
        m = _build_hier()
        _, value, _ = _forward_hier(m, batch=4)
        assert value.shape == (4,)

    def test_logits_batch_dimension(self):
        m = _build_hier()
        logits, _, _ = _forward_hier(m, batch=8)
        assert logits.shape == (8, 3)

    def test_logits_are_finite(self):
        m = _build_hier()
        logits, _, _ = _forward_hier(m)
        assert torch.isfinite(logits).all()

    def test_value_is_finite(self):
        m = _build_hier()
        _, value, _ = _forward_hier(m)
        assert torch.isfinite(value).all()

    def test_state_is_empty_list(self):
        m = _build_hier()
        _, _, state = _forward_hier(m)
        assert state == []

    def test_num_outputs_6(self):
        m = _build_hier(num_outputs=6)
        logits, _, _ = _forward_hier(m)
        assert logits.shape == (1, 6)

    def test_custom_conv_channels(self):
        m = _build_hier(conv_channels=[16, 32])
        logits, _, _ = _forward_hier(m)
        assert logits.shape == (1, 3)

    def test_custom_fc_hiddens(self):
        m = _build_hier(fc_hiddens=[128, 64])
        logits, _, _ = _forward_hier(m)
        assert logits.shape == (1, 3)

    def test_max_pool_variant(self):
        m = _build_hier(pool_type="max")
        logits, _, _ = _forward_hier(m)
        assert logits.shape == (1, 3)

    def test_single_radius_raises(self):
        with pytest.raises(ValueError, match="at least 2 radii"):
            _build_hier(radii=[2])

    def test_flat_size_r1_r4(self):
        """flat input size = 3^3 + 9^3 = 27 + 729 = 756."""
        flat = sum((2 * r + 1) ** 3 for r in [1, 4])
        assert flat == 756

    def test_flat_size_r1_r2_r4(self):
        """Three radii: 3^3 + 5^3 + 9^3 = 27 + 125 + 729 = 881."""
        flat = sum((2 * r + 1) ** 3 for r in [1, 2, 4])
        assert flat == 881

    def test_n_max_is_largest_radius_size(self):
        m = _build_hier(radii=[1, 3])
        assert m._n_max == 2 * 3 + 1  # 7

    def test_n_channels_equals_num_radii(self):
        m = _build_hier(radii=[1, 2, 4])
        # First Conv3d layer input channels = 3
        first_layer = m._conv[0]
        assert first_layer.in_channels == 3

    def test_two_radii_two_channels(self):
        m = _build_hier(radii=[1, 4])
        first_layer = m._conv[0]
        assert first_layer.in_channels == 2

    def test_padding_preserves_spatial_symmetry(self):
        """Small box (r=1, 3^3) should be padded by 3 on each side to reach 9^3."""
        m = _build_hier(radii=[1, 4])
        # r=1: n=3, n_max=9, pad=3 on each side
        assert (m._n_max - m._sizes[0]) // 2 == 3

    def test_gradient_flows(self):
        m = _build_hier()
        flat_size = sum((2 * r + 1) ** 3 for r in [1, 4])
        obs = {
            "steps_remaining": torch.rand(1, 1, requires_grad=False),
            "voxel_count":     torch.rand(1, 1),
            "cursor_pos":      torch.rand(1, 3),
            "local_grid":      torch.rand(1, flat_size, requires_grad=True),
        }
        logits, _ = m.forward({"obs": obs}, [], torch.tensor([1]))
        logits.sum().backward()
        assert obs["local_grid"].grad is not None


# ---------------------------------------------------------------------------
# Registration update test
# ---------------------------------------------------------------------------

class TestHierarchicalModelRegistration:
    def test_hierarchical_key_registered(self):
        from ray.rllib.models.catalog import RLLIB_MODEL, _global_registry
        from theseo_anysearch.rllib.models.cnn import register_voxel_cnn_models
        register_voxel_cnn_models()
        assert _global_registry.contains(RLLIB_MODEL, "voxel_hierarchical_box_3d_cnn")

    def test_all_three_keys_registered(self):
        from ray.rllib.models.catalog import RLLIB_MODEL, _global_registry
        from theseo_anysearch.rllib.models.cnn import register_voxel_cnn_models
        register_voxel_cnn_models()
        for key in ("voxel_box_2d_cnn", "voxel_box_3d_cnn", "voxel_hierarchical_box_3d_cnn"):
            assert _global_registry.contains(RLLIB_MODEL, key)


# ---------------------------------------------------------------------------
# VoxelEnv obs mode tests — hierarchical_box
# ---------------------------------------------------------------------------

class TestVoxelEnvHierarchicalBoxObsMode:
    def test_observation_space_flat_size_r1_r4(self):
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
        env = VoxelEnv.__new__(VoxelEnv)
        env._config = {"obs_mode": "hierarchical_box", "box_radii": [1, 4]}
        env._rust_env = None
        sp = env._observation_space()
        assert sp["local_grid"].shape == (756,)   # 3^3 + 9^3

    def test_observation_space_flat_size_r1_r2_r4(self):
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
        env = VoxelEnv.__new__(VoxelEnv)
        env._config = {"obs_mode": "hierarchical_box", "box_radii": [1, 2, 4]}
        env._rust_env = None
        sp = env._observation_space()
        assert sp["local_grid"].shape == (881,)

    def test_observation_space_default_radii(self):
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
        env = VoxelEnv.__new__(VoxelEnv)
        env._config = {"obs_mode": "hierarchical_box"}
        env._rust_env = None
        sp = env._observation_space()
        # Default radii [1, 4] → 27 + 729 = 756
        assert sp["local_grid"].shape == (756,)

    def test_observation_space_has_cursor_pos(self):
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
        env = VoxelEnv.__new__(VoxelEnv)
        env._config = {"obs_mode": "hierarchical_box"}
        env._rust_env = None
        sp = env._observation_space()
        assert "cursor_pos" in sp.spaces

    def test_obs_to_numpy_concatenates_segments(self):
        """_obs_to_numpy should concatenate box_obs(r) calls in radius order."""
        from unittest.mock import MagicMock
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

        env = VoxelEnv.__new__(VoxelEnv)
        env._config = {
            "obs_mode": "hierarchical_box",
            "box_radii": [1, 4],
            "max_steps": 10,
        }

        mock_rust = MagicMock()
        mock_rust.box_obs.side_effect = lambda r: [0.1] * (2 * r + 1) ** 3
        mock_rust.cursor_pos.return_value = (5, 5, 5)
        env._rust_env = mock_rust
        env._init_obs_cache(env._config)

        mock_obs = MagicMock()
        mock_obs.steps_remaining = 5
        mock_obs.filled = 3

        result = env._obs_to_numpy(mock_obs)
        assert "local_grid" in result
        assert result["local_grid"].shape == (756,)
        # box_obs called once per radius
        assert mock_rust.box_obs.call_count == 2

    def test_obs_to_numpy_unknown_mode_raises(self):
        from unittest.mock import MagicMock as MM
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
        env = VoxelEnv.__new__(VoxelEnv)
        env._config = {"obs_mode": "magic", "max_steps": 10}
        env._rust_env = MM()
        env._rust_env.cursor_pos.return_value = (1, 1, 1)
        env._init_obs_cache(env._config)
        mock_obs = MM()
        mock_obs.steps_remaining = 1
        mock_obs.filled = 0
        with pytest.raises(ValueError, match="hierarchical_box"):
            env._obs_to_numpy(mock_obs)

    def test_env_config_accepts_box_radii(self):
        from theseo_anysearch.models import EnvConfig
        from pathlib import Path
        cfg = EnvConfig(
            stl_path=Path("/tmp/x.stl"),
            obs_mode="hierarchical_box",
            box_radii=[1, 4],
        )
        assert cfg.box_radii == [1, 4]
        assert cfg.obs_mode == "hierarchical_box"
