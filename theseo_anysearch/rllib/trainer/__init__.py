"""Training lifecycle interfaces and compatibility exports."""

from __future__ import annotations

from typing import Any

from theseo_anysearch.rllib.trainer.base import BaseTrainer
from theseo_anysearch.rllib.trainer.results import RllibTrainResult, TrainResult
from theseo_anysearch.rllib.trainer.trainer import Trainer

_ALGORITHM_EXPORTS = {
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
    """Load a legacy algorithm trainer export on demand.

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
    target = _ALGORITHM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)


__all__ = [
    "BaseTrainer",
    "DDPGTrainer",
    "DQNTrainer",
    "MultiAgentVoxelPPOTrainer",
    "PPOTrainer",
    "RainbowTrainer",
    "RllibTrainResult",
    "SACTrainer",
    "TD3Trainer",
    "TrainResult",
    "Trainer",
]