from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


def _build_rllib_model_dict(model_cfg: Any) -> dict:
    """Return the RLlib model dict for .training(model=...) from a ModelConfig."""
    if model_cfg.custom_model:
        from theseo_anysearch.rllib.models import register_voxel_cnn_models
        register_voxel_cnn_models()
        return {
            "custom_model": model_cfg.custom_model,
            "custom_model_config": model_cfg.custom_model_config or {},
        }
    return {
        "fcnet_hiddens": model_cfg.hidden_sizes,
        "fcnet_activation": model_cfg.activation,
    }


def _build_conv2d_stack(
    in_channels: int, channels: list[int]
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    c_in = in_channels
    for c_out in channels:
        layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU()]
        c_in = c_out
    return nn.Sequential(*layers), c_in


def _build_conv3d_stack(
    channels: list[int], in_channels: int = 1
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    c_in = in_channels
    for c_out in channels:
        layers += [nn.Conv3d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU()]
        c_in = c_out
    return nn.Sequential(*layers), c_in


def _build_fc_stack(in_features: int, hiddens: list[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = in_features
    for h in hiddens:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers), d


class VoxelBox2DCNN(TorchModelV2, nn.Module):
    """
    2D CNN custom model for VoxelEnv box observations.

    Reshapes the flat local_grid (N³,) into (C=N, H=N, W=N) treating the z-axis
    as channels, applies Conv2d layers, global-average-pools to (C_last,), then
    concatenates scalar features (steps_remaining, voxel_count, cursor_pos) before
    the FC head and policy/value output heads.

    Conforms to the RLlib legacy TorchModelV2 interface.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {}) or {}
        box_radius: int = cfg.get("box_radius", 2)
        conv_channels: list[int] = cfg.get("conv_channels", [32, 64, 64])
        fc_hiddens: list[int] = cfg.get("fc_hiddens", [256])
        pool_type: str = cfg.get("pool_type", "avg")

        self._n = 2 * box_radius + 1
        n_scalars = 5  # steps_remaining(1) + voxel_count(1) + cursor_pos(3)

        self._conv, last_c = _build_conv2d_stack(self._n, conv_channels)
        self._pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if pool_type == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self._fc, fc_out = _build_fc_stack(last_c + n_scalars, fc_hiddens)
        self._policy_head = nn.Linear(fc_out, num_outputs)
        self._value_head = nn.Linear(fc_out, 1)
        self._value_out: torch.Tensor | None = None

    def forward(
        self,
        input_dict: dict,
        state: list,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list]:
        obs = input_dict["obs"]
        grid = obs["local_grid"]  # (B, N³)
        b = grid.shape[0]
        # Reshape to (B, N, N, N) = (B, x, y, z), then permute z → channels: (B, z, x, y)
        grid_3d = grid.view(b, self._n, self._n, self._n).permute(0, 3, 1, 2)
        feat = self._conv(grid_3d)       # (B, last_c, N, N)
        feat = self._pool(feat)          # (B, last_c, 1, 1)
        feat = feat.view(b, -1)          # (B, last_c)

        scalars = torch.cat([
            obs["steps_remaining"],
            obs["voxel_count"],
            obs["cursor_pos"],
        ], dim=1)  # (B, 5)
        x = torch.cat([feat, scalars], dim=1)  # (B, last_c + 5)
        x = self._fc(x)                        # (B, fc_out)

        self._value_out = self._value_head(x)  # (B, 1)
        return self._policy_head(x), state

    def value_function(self) -> torch.Tensor:
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(1)


class VoxelBox3DCNN(TorchModelV2, nn.Module):
    """
    3D CNN custom model for VoxelEnv box observations.

    Reshapes the flat local_grid (N³,) into a single-channel 3D volume (1, N, N, N),
    applies Conv3d layers, global-average-pools to (C_last,), then concatenates
    scalar features before the FC head and policy/value output heads.

    Conforms to the RLlib legacy TorchModelV2 interface.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {}) or {}
        box_radius: int = cfg.get("box_radius", 2)
        conv_channels: list[int] = cfg.get("conv_channels", [32, 64, 64])
        fc_hiddens: list[int] = cfg.get("fc_hiddens", [256])
        pool_type: str = cfg.get("pool_type", "avg")

        self._n = 2 * box_radius + 1
        n_scalars = 5

        self._conv, last_c = _build_conv3d_stack(conv_channels)
        self._pool = (
            nn.AdaptiveAvgPool3d((1, 1, 1))
            if pool_type == "avg"
            else nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self._fc, fc_out = _build_fc_stack(last_c + n_scalars, fc_hiddens)
        self._policy_head = nn.Linear(fc_out, num_outputs)
        self._value_head = nn.Linear(fc_out, 1)
        self._value_out: torch.Tensor | None = None

    def forward(
        self,
        input_dict: dict,
        state: list,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list]:
        obs = input_dict["obs"]
        grid = obs["local_grid"]  # (B, N³)
        b = grid.shape[0]
        # Reshape to (B, 1, N, N, N)
        grid_3d = grid.view(b, 1, self._n, self._n, self._n)
        feat = self._conv(grid_3d)       # (B, last_c, N, N, N)
        feat = self._pool(feat)          # (B, last_c, 1, 1, 1)
        feat = feat.view(b, -1)          # (B, last_c)

        scalars = torch.cat([
            obs["steps_remaining"],
            obs["voxel_count"],
            obs["cursor_pos"],
        ], dim=1)  # (B, 5)
        x = torch.cat([feat, scalars], dim=1)
        x = self._fc(x)

        self._value_out = self._value_head(x)
        return self._policy_head(x), state

    def value_function(self) -> torch.Tensor:
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(1)


class VoxelHierarchicalBox3DCNN(TorchModelV2, nn.Module):
    """
    3D CNN for hierarchical (multi-scale) box observations.

    Two or more box observations at different radii are concatenated in the
    observation as a single flat ``local_grid`` key of shape
    ``(sum_i (2*r_i+1)^3,)``.  Each segment is reshaped to its own 3D volume
    and then centre-padded to the largest radius size ``(N_max, N_max, N_max)``.
    The padded volumes are stacked as channels — one channel per radius — giving
    a tensor of shape ``(B, n_radii, N_max, N_max, N_max)`` which is processed
    by 3D Conv layers.

    Config keys (``custom_model_config``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    box_radii : list[int]
        Radii to use, in the same order they appear in the observation.
        Defaults to ``[1, 4]`` (tight + coarse).
    conv_channels : list[int]
        Output channels for each Conv3d layer.  Defaults to ``[32, 64, 64]``.
    fc_hiddens : list[int]
        Hidden layer widths for the FC trunk.  Defaults to ``[256]``.
    pool_type : str
        ``"avg"`` (default) or ``"max"``.

    Example YAML::

        model_config:
          custom_model: voxel_hierarchical_box_3d_cnn
          custom_model_config:
            box_radii: [1, 4]
            conv_channels: [32, 64, 64]
            fc_hiddens: [256]
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {}) or {}
        radii: list[int] = cfg.get("box_radii", [1, 4])
        conv_channels: list[int] = cfg.get("conv_channels", [32, 64, 64])
        fc_hiddens: list[int] = cfg.get("fc_hiddens", [256])
        pool_type: str = cfg.get("pool_type", "avg")

        if len(radii) < 2:
            raise ValueError("VoxelHierarchicalBox3DCNN requires at least 2 radii.")

        self._radii = radii
        self._sizes = [2 * r + 1 for r in radii]   # n_i for each radius
        self._n_max = max(self._sizes)               # spatial size after padding
        n_channels = len(radii)                      # one channel per radius
        n_scalars = 5                                # steps_remaining + voxel_count + cursor_pos

        self._conv, last_c = _build_conv3d_stack(conv_channels, in_channels=n_channels)
        self._pool = (
            nn.AdaptiveAvgPool3d((1, 1, 1))
            if pool_type == "avg"
            else nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self._fc, fc_out = _build_fc_stack(last_c + n_scalars, fc_hiddens)
        self._policy_head = nn.Linear(fc_out, num_outputs)
        self._value_head = nn.Linear(fc_out, 1)
        self._value_out: torch.Tensor | None = None

    def forward(
        self,
        input_dict: dict,
        state: list,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list]:
        obs = input_dict["obs"]
        grid = obs["local_grid"]   # (B, sum_i n_i^3)
        b = grid.shape[0]

        # Split flat obs into per-radius segments, reshape and centre-pad to N_max
        volumes: list[torch.Tensor] = []
        offset = 0
        for n_i in self._sizes:
            seg = grid[:, offset : offset + n_i ** 3]       # (B, n_i^3)
            vol = seg.view(b, n_i, n_i, n_i)                # (B, n_i, n_i, n_i)
            pad = (self._n_max - n_i) // 2
            if pad > 0:
                # F.pad expects (left, right) per dim, innermost first: x, y, z
                vol = F.pad(vol, [pad, pad, pad, pad, pad, pad])
            volumes.append(vol)
            offset += n_i ** 3

        # Stack as channels: (B, n_radii, N_max, N_max, N_max)
        x = torch.stack(volumes, dim=1)

        feat = self._conv(x)    # (B, last_c, N_max, N_max, N_max)
        feat = self._pool(feat) # (B, last_c, 1, 1, 1)
        feat = feat.view(b, -1) # (B, last_c)

        scalars = torch.cat([
            obs["steps_remaining"],
            obs["voxel_count"],
            obs["cursor_pos"],
        ], dim=1)  # (B, 5)
        x = torch.cat([feat, scalars], dim=1)
        x = self._fc(x)

        self._value_out = self._value_head(x)
        return self._policy_head(x), state

    def value_function(self) -> torch.Tensor:
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(1)


def register_voxel_cnn_models() -> None:
    """Register all VoxelCNN custom models with RLlib's ModelCatalog."""
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_model("voxel_box_2d_cnn", VoxelBox2DCNN)
    ModelCatalog.register_custom_model("voxel_box_3d_cnn", VoxelBox3DCNN)
    ModelCatalog.register_custom_model(
        "voxel_hierarchical_box_3d_cnn", VoxelHierarchicalBox3DCNN
    )
