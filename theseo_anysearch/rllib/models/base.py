"""Shared RLlib model helpers and voxel model base classes.

This module defines the shared ownership points for RLlib-facing Python models:
abstract observation encoders, translation from project model configs to RLlib
model dictionaries, and a common ``TorchModelV2`` base for voxel policy/value
models.

Examples
--------
Build the RLlib model configuration for a standard fully connected model::

    from theseo_anysearch.rllib.models.base import build_rllib_model_dict
    from theseo_anysearch.settings import ModelConfig

    cfg = ModelConfig(hidden_sizes=[256, 256], activation="relu")
    model = build_rllib_model_dict(cfg)

Build the RLlib model configuration for a registered custom CNN model::

    from theseo_anysearch.rllib.models.base import build_rllib_model_dict
    from theseo_anysearch.settings import ModelConfig

    cfg = ModelConfig(
        custom_model="voxel_box_3d_cnn",
        custom_model_config={"box_radius": 2},
    )
    model = build_rllib_model_dict(cfg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class BaseRLModule(ABC):
    """Abstract base for project-specific RL modules.

    Methods
    -------
    encode_observation
        Convert a raw observation dictionary into a latent representation.
    """

    @abstractmethod
    def encode_observation(self, obs: dict) -> object:
        """Encode an observation dictionary into a latent representation.

        Parameters
        ----------
        obs : dict
            Raw observation mapping received from the environment.

        Returns
        -------
        object
            Encoded representation consumed by downstream policy or value code.
        """


def build_rllib_model_dict(model_cfg: Any) -> dict:
    """Build the RLlib ``model`` config dictionary from a project model config.

    Parameters
    ----------
    model_cfg : Any
        Project model configuration object with either standard FC settings or
        a custom model registration reference.

    Returns
    -------
    dict
        Dictionary passed into RLlib's ``training(model=...)`` configuration.
    """
    if model_cfg.custom_model:
        from theseo_anysearch.rllib.models.cnn import register_voxel_cnn_models

        register_voxel_cnn_models()
        return {
            "custom_model": model_cfg.custom_model,
            "custom_model_config": model_cfg.custom_model_config or {},
        }
    return {
        "fcnet_hiddens": model_cfg.hidden_sizes,
        "fcnet_activation": model_cfg.activation,
    }


def build_rllib_rl_module_model_config(model_cfg: Any) -> Any:
    """Translate a project model config for RLlib's RLModule stack.

    Parameters
    ----------
    model_cfg : Any
        Validated AnySearch model settings.

    Returns
    -------
    DefaultModelConfig
        RLlib's configuration for its algorithm-specific default RLModule.

    Raises
    ------
    ValueError
        If the configuration references a legacy ``TorchModelV2`` model.
    """
    if model_cfg.custom_model:
        raise ValueError(
            f"custom model '{model_cfg.custom_model}' still uses TorchModelV2; "
            "migrate it to RLModule before enabling the new RLlib API stack"
        )

    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    return DefaultModelConfig(
        fcnet_hiddens=list(model_cfg.hidden_sizes),
        fcnet_activation=model_cfg.activation,
    )


class BaseVoxelTorchModel(TorchModelV2, nn.Module):
    """Shared ``TorchModelV2`` base for voxel policy/value models.

    This base owns the repeated head-building and scalar-feature handling used
    by the concrete voxel CNN implementations.

    Parameters
    ----------
    obs_space : Any
        RLlib observation space for the model.
    action_space : Any
        RLlib action space for the model.
    num_outputs : int
        Number of policy output logits produced by the model.
    model_config : dict
        RLlib model configuration dictionary.
    name : str
        RLlib model instance name.
    """

    _spatial_keys = frozenset({"local_grid"})

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._value_out: torch.Tensor | None = None
        structured_space = getattr(obs_space, "original_space", obs_space)
        spaces = getattr(structured_space, "spaces", None)
        if spaces is None:
            raise ValueError("voxel CNN models require a Dict observation space")
        self._auxiliary_keys = tuple(
            key for key in spaces if key not in self._spatial_keys
        )
        self._n_auxiliary = sum(
            int(space.shape[0]) for key, space in spaces.items()
            if key in self._auxiliary_keys
        )

    @staticmethod
    def _custom_model_config(model_config: dict) -> dict[str, Any]:
        """Return the custom model config block as a normalized dictionary."""
        return model_config.get("custom_model_config", {}) or {}

    @classmethod
    def _build_fc_stack(cls, in_features: int, hiddens: list[int]) -> tuple[nn.Sequential, int]:
        """Build a ReLU MLP stack and return the stack with its output width."""
        layers: list[nn.Module] = []
        d = in_features
        for h in hiddens:
            layers.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        return nn.Sequential(*layers), d

    @classmethod
    def _build_conv2d_stack(
        cls, in_channels: int, channels: list[int]
    ) -> tuple[nn.Sequential, int]:
        """Build a 2D convolution stack and return the stack with its last channel count."""
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU()])
            c_in = c_out
        return nn.Sequential(*layers), c_in

    @classmethod
    def _build_conv3d_stack(
        cls, channels: list[int], in_channels: int = 1
    ) -> tuple[nn.Sequential, int]:
        """Build a 3D convolution stack and return the stack with its last channel count."""
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([nn.Conv3d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU()])
            c_in = c_out
        return nn.Sequential(*layers), c_in

    def _build_heads(self, in_features: int, fc_hiddens: list[int], num_outputs: int) -> None:
        """Build the shared FC trunk and policy/value output heads."""
        self._fc, fc_out = self._build_fc_stack(in_features, fc_hiddens)
        self._policy_head = nn.Linear(fc_out, num_outputs)
        self._value_head = nn.Linear(fc_out, 1)

    def _auxiliary_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate all non-spatial fields declared by the observation space."""
        if not self._auxiliary_keys:
            grid = obs["local_grid"]
            return grid.new_zeros((grid.shape[0], 0))
        return torch.cat(
            [obs[key].reshape(obs[key].shape[0], -1) for key in self._auxiliary_keys],
            dim=1,
        )

    def _forward_heads(self, features: torch.Tensor, state: list) -> tuple[torch.Tensor, list]:
        """Run the shared FC trunk and produce policy logits plus cached value output."""
        x = self._fc(features)
        self._value_out = self._value_head(x)
        return self._policy_head(x), state

    def value_function(self) -> torch.Tensor:
        """Return the cached scalar value prediction from the most recent forward pass.

        Returns
        -------
        torch.Tensor
            Value predictions with shape ``(batch,)``.
        """
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(1)
