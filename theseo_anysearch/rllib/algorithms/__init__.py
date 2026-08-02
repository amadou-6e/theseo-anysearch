"""Algorithm-specific RLlib settings and construction adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DDPGTrainer": ("theseo_anysearch.rllib.algorithms.ddpg", "DDPGTrainer"),
    "DQNTrainer": ("theseo_anysearch.rllib.algorithms.dqn", "DQNTrainer"),
    "MultiAgentVoxelPPOTrainer": (
        "theseo_anysearch.rllib.algorithms.multi_voxel_ppo",
        "MultiAgentVoxelPPOTrainer",
    ),
    "PPOTrainer": ("theseo_anysearch.rllib.algorithms.ppo", "PPOTrainer"),
    "RainbowTrainer": ("theseo_anysearch.rllib.algorithms.rainbow", "RainbowTrainer"),
    "SACTrainer": ("theseo_anysearch.rllib.algorithms.sac", "SACTrainer"),
    "TD3Trainer": ("theseo_anysearch.rllib.algorithms.td3", "TD3Trainer"),
}


def __getattr__(name: str) -> Any:
    """Load an algorithm adapter on demand.

    Parameters
    ----------
    name : str
        Public attribute requested from the package.

    Returns
    -------
    Any
        Requested algorithm adapter class.

    Raises
    ------
    AttributeError
        If the package does not export name.
    """
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)


__all__ = list(_EXPORTS)