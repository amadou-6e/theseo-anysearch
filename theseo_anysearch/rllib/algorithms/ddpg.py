"""DDPG trainer integration and configuration wiring."""

from __future__ import annotations

from typing import Any

from theseo_anysearch.rllib.trainer.trainer import Trainer


class DDPGTrainer(Trainer):
    """
    DDPG trainer backed by ray.rllib.algorithms.ddpg.DDPG (legacy API stack).

    NOT YET USABLE: DDPG requires a continuous (Box) action space environment.
    The current VoxelEnv (Discrete(3)) and SurfaceEnv (Discrete(1)) are not
    compatible. Implement a continuous-action environment before enabling DDPG.
    """

    algorithm_name = "ddpg"

    def _build_algorithm(self) -> Any:
        raise NotImplementedError(
            "DDPG requires a continuous (Box) action space environment. "
            "VoxelEnv (Discrete(3)) and SurfaceEnv (Discrete(1)) are not compatible. "
            "Add a Box-action environment and wire it here before using DDPG."
        )
