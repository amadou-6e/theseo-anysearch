"""Explicit registry for built-in RLlib algorithm adapters."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theseo_anysearch.rllib.trainer.trainer import Trainer


_TRAINER_TYPES = {
    "ddpg": ("theseo_anysearch.rllib.algorithms.ddpg", "DDPGTrainer"),
    "dqn": ("theseo_anysearch.rllib.algorithms.dqn", "DQNTrainer"),
    "multi_agent_voxel_ppo": (
        "theseo_anysearch.rllib.algorithms.ppo",
        "MultiAgentVoxelPPOTrainer",
    ),
    "ppo": ("theseo_anysearch.rllib.algorithms.ppo", "PPOTrainer"),
    "rainbow": ("theseo_anysearch.rllib.algorithms.rainbow", "RainbowTrainer"),
    "sac": ("theseo_anysearch.rllib.algorithms.sac", "SACTrainer"),
    "td3": ("theseo_anysearch.rllib.algorithms.td3", "TD3Trainer"),
}


def get_trainer_class(algorithm: str) -> type[Trainer]:
    """Resolve a YAML algorithm name to its trainer adapter.

    Parameters
    ----------
    algorithm : str
        Algorithm name from training settings.

    Returns
    -------
    type[Trainer]
        Canonical algorithm adapter class.

    Raises
    ------
    NotImplementedError
        If no built-in adapter exists for the requested algorithm.
    """
    target = _TRAINER_TYPES.get(algorithm.lower())
    if target is None:
        raise NotImplementedError(
            f"No Trainer registered for algorithm '{algorithm}'."
        )
    module_name, class_name = target
    module = import_module(module_name)
    return getattr(module, class_name)


def registered_algorithms() -> tuple[str, ...]:
    """Return the registered built-in algorithm names.

    Returns
    -------
    tuple[str, ...]
        Sorted YAML-selectable algorithm names.
    """
    return tuple(sorted(_TRAINER_TYPES))
