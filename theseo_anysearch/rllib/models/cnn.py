"""CNN-based RLlib models for voxel observations.

This module contains the concrete voxel CNN variants used by RLlib custom
models, including standard 2D and 3D encoders, a hierarchical multi-radius
3D model, and a variant backed by a pretrained garden encoder.

Examples
--------
Register the voxel CNN models with RLlib::

    from theseo_anysearch.rllib.models.cnn import register_voxel_cnn_models

    register_voxel_cnn_models()

Reference a custom voxel CNN model from configuration::

    model_config:
      custom_model: voxel_box_3d_cnn
      custom_model_config:
        box_radius: 2
        conv_channels: [32, 64, 64]
        fc_hiddens: [256]
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from theseo_anysearch.rllib.models.base import BaseVoxelTorchModel


class VoxelBox2DCNN(BaseVoxelTorchModel):
    """2D CNN model for single-radius voxel box observations.

    The model interprets the voxel box as a 2D feature map stack by treating the
    z-axis as channels, then combines the spatial convolution map with all
    non-grid observation features before producing policy and value outputs.

    Parameters
    ----------
    obs_space : Any
        RLlib observation space.
    action_space : Any
        RLlib action space.
    num_outputs : int
        Number of policy output logits.
    model_config : dict
        RLlib model config containing ``custom_model_config`` overrides.
    name : str
        RLlib model instance name.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        cfg = self._custom_model_config(model_config)
        box_radius: int = cfg.get("box_radius", 2)
        conv_channels: list[int] = cfg.get("conv_channels", [32, 64, 64])
        fc_hiddens: list[int] = cfg.get("fc_hiddens", [256])
        self._n = 2 * box_radius + 1

        self._conv, last_c = self._build_conv2d_stack(self._n, conv_channels)
        self._build_heads(
            last_c * self._n * self._n + self._n_auxiliary,
            fc_hiddens,
            num_outputs,
        )

    def forward(
        self,
        input_dict: dict,
        state: list,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list]:
        """Compute policy logits for a batch of voxel observations.

        Parameters
        ----------
        input_dict : dict
            RLlib input dictionary containing the observation batch.
        state : list
            Recurrent state list. This model does not modify it.
        seq_lens : torch.Tensor
            Sequence lengths provided by RLlib.

        Returns
        -------
        tuple[torch.Tensor, list]
            Policy logits and the unchanged recurrent state.
        """
        obs = input_dict["obs"]
        grid = obs["local_grid"]
        batch = grid.shape[0]
        grid_3d = grid.view(batch, self._n, self._n, self._n).permute(0, 3, 1, 2)
        feat = self._conv(grid_3d)
        feat = feat.view(batch, -1)

        auxiliary = self._auxiliary_features(obs)
        features = torch.cat([feat, auxiliary], dim=1)
        return self._forward_heads(features, state)


class VoxelBox3DCNN(BaseVoxelTorchModel):
    """3D CNN model for single-radius voxel box observations.

    The model reshapes the voxel box into a single-channel 3D volume, extracts
    spatial convolution features, and combines them with all non-grid features
    before producing policy and value outputs.

    Parameters
    ----------
    obs_space : Any
        RLlib observation space.
    action_space : Any
        RLlib action space.
    num_outputs : int
        Number of policy output logits.
    model_config : dict
        RLlib model config containing ``custom_model_config`` overrides.
    name : str
        RLlib model instance name.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        cfg = self._custom_model_config(model_config)
        box_radius: int = cfg.get("box_radius", 2)
        conv_channels: list[int] = cfg.get("conv_channels", [32, 64, 64])
        fc_hiddens: list[int] = cfg.get("fc_hiddens", [256])
        self._n = 2 * box_radius + 1

        self._conv, last_c = self._build_conv3d_stack(conv_channels)
        self._build_heads(
            last_c * self._n**3 + self._n_auxiliary,
            fc_hiddens,
            num_outputs,
        )

    def forward(
        self,
        input_dict: dict,
        state: list,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list]:
        """Compute policy logits for a batch of voxel observations.

        Parameters
        ----------
        input_dict : dict
            RLlib input dictionary containing the observation batch.
        state : list
            Recurrent state list. This model does not modify it.
        seq_lens : torch.Tensor
            Sequence lengths provided by RLlib.

        Returns
        -------
        tuple[torch.Tensor, list]
            Policy logits and the unchanged recurrent state.
        """
        obs = input_dict["obs"]
        grid = obs["local_grid"]
        batch = grid.shape[0]
        grid_3d = grid.view(batch, 1, self._n, self._n, self._n)
        feat = self._conv(grid_3d)
        feat = feat.view(batch, -1)

        auxiliary = self._auxiliary_features(obs)
        features = torch.cat([feat, auxiliary], dim=1)
        return self._forward_heads(features, state)


class VoxelHierarchicalBox3DCNN(BaseVoxelTorchModel):
    """3D CNN model for hierarchical multi-radius voxel observations.

    The model splits a flattened multi-radius observation into separate voxel
    volumes, center-pads them to a common size, stacks them as channels, and
    then applies a shared 3D convolution trunk before the policy/value heads.

    Parameters
    ----------
    obs_space : Any
        RLlib observation space.
    action_space : Any
        RLlib action space.
    num_outputs : int
        Number of policy output logits.
    model_config : dict
        RLlib model config containing ``custom_model_config`` overrides.
    name : str
        RLlib model instance name.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        cfg = self._custom_model_config(model_config)
        radii: list[int] = cfg.get("box_radii", [1, 4])
        conv_channels: list[int] = cfg.get("conv_channels", [32, 64, 64])
        fc_hiddens: list[int] = cfg.get("fc_hiddens", [256])
        if len(radii) < 2:
            raise ValueError("VoxelHierarchicalBox3DCNN requires at least 2 radii.")

        self._radii = radii
        self._sizes = [2 * r + 1 for r in radii]
        self._n_max = max(self._sizes)
        n_channels = len(radii)

        self._conv, last_c = self._build_conv3d_stack(conv_channels, in_channels=n_channels)
        self._build_heads(
            last_c * self._n_max**3 + self._n_auxiliary,
            fc_hiddens,
            num_outputs,
        )

    def forward(
        self,
        input_dict: dict,
        state: list,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list]:
        """Compute policy logits for a batch of hierarchical voxel observations.

        Parameters
        ----------
        input_dict : dict
            RLlib input dictionary containing the observation batch.
        state : list
            Recurrent state list. This model does not modify it.
        seq_lens : torch.Tensor
            Sequence lengths provided by RLlib.

        Returns
        -------
        tuple[torch.Tensor, list]
            Policy logits and the unchanged recurrent state.
        """
        obs = input_dict["obs"]
        grid = obs["local_grid"]
        batch = grid.shape[0]

        volumes: list[torch.Tensor] = []
        offset = 0
        for size in self._sizes:
            segment = grid[:, offset : offset + size**3]
            volume = segment.view(batch, size, size, size)
            pad = (self._n_max - size) // 2
            if pad > 0:
                volume = F.pad(volume, [pad, pad, pad, pad, pad, pad])
            volumes.append(volume)
            offset += size**3

        stacked = torch.stack(volumes, dim=1)
        feat = self._conv(stacked)
        feat = feat.view(batch, -1)

        auxiliary = self._auxiliary_features(obs)
        features = torch.cat([feat, auxiliary], dim=1)
        return self._forward_heads(features, state)


class VoxelBox3DCNNPretrained(BaseVoxelTorchModel):
    """3D CNN policy head backed by a pretrained garden encoder.

    This model loads a garden encoder checkpoint, optionally freezes it, and
    uses the resulting latent vector together with all non-grid observation
    features to produce policy and value outputs.

    Parameters
    ----------
    obs_space : Any
        RLlib observation space.
    action_space : Any
        RLlib action space.
    num_outputs : int
        Number of policy output logits.
    model_config : dict
        RLlib model config containing ``custom_model_config`` overrides.
    name : str
        RLlib model instance name.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        cfg = self._custom_model_config(model_config)
        encoder_ref: str = cfg["pretrained_encoder"]
        freeze: bool = cfg.get("freeze_encoder", True)
        fc_hiddens: list[int] = cfg.get("fc_hiddens", [256, 256])

        from theseo_anysearch.garden.models.ae import VoxelEncoder3D
        from theseo_anysearch.garden.store import GardenStore
        import json

        model_dir, _ = GardenStore.resolve_ref(encoder_ref)
        meta = json.loads((model_dir / "meta.json").read_text())
        box_radius: int = meta["box_radius"]
        latent_dim: int = meta["latent_dim"]
        conv_channels: list[int] = meta["conv_channels"]

        self._n = 2 * box_radius + 1
        encoder = VoxelEncoder3D(
            n=self._n,
            channels=conv_channels,
            latent_dim=latent_dim,
        )
        state = GardenStore(model_dir.parent).load_encoder_state(model_dir.name)
        encoder.load_state_dict(state)

        if freeze:
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            encoder.eval()

        self._encoder = encoder
        self._freeze = freeze
        self._build_heads(latent_dim + self._n_auxiliary, fc_hiddens, num_outputs)

    def forward(
        self,
        input_dict: dict,
        state: list,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list]:
        """Compute policy logits using the pretrained encoder and shared heads.

        Parameters
        ----------
        input_dict : dict
            RLlib input dictionary containing the observation batch.
        state : list
            Recurrent state list. This model does not modify it.
        seq_lens : torch.Tensor
            Sequence lengths provided by RLlib.

        Returns
        -------
        tuple[torch.Tensor, list]
            Policy logits and the unchanged recurrent state.
        """
        obs = input_dict["obs"]
        grid = obs["local_grid"]
        batch = grid.shape[0]
        grid_3d = grid.view(batch, self._n, self._n, self._n)

        if self._freeze:
            with torch.no_grad():
                latent = self._encoder(grid_3d)
        else:
            latent = self._encoder(grid_3d)

        auxiliary = self._auxiliary_features(obs)
        features = torch.cat([latent, auxiliary], dim=1)
        return self._forward_heads(features, state)


def register_voxel_cnn_models() -> None:
    """Register the voxel CNN custom models with RLlib's ``ModelCatalog``.

    Returns
    -------
    None
        This function mutates RLlib's global model registry in place.
    """
    from ray.rllib.models import ModelCatalog

    ModelCatalog.register_custom_model("voxel_box_2d_cnn", VoxelBox2DCNN)
    ModelCatalog.register_custom_model("voxel_box_3d_cnn", VoxelBox3DCNN)
    ModelCatalog.register_custom_model(
        "voxel_hierarchical_box_3d_cnn", VoxelHierarchicalBox3DCNN
    )
    ModelCatalog.register_custom_model(
        "voxel_box_3d_cnn_pretrained", VoxelBox3DCNNPretrained
    )
