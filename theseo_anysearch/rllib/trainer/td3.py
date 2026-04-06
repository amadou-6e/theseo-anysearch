from __future__ import annotations

from typing import Any

from theseo_anysearch.rllib.trainer.base import Trainer


class TD3Trainer(Trainer):
    """
    TD3 trainer backed by ray.rllib.algorithms.td3.TD3 (legacy API stack).

    NOT YET USABLE: TD3 requires a continuous (Box) action space environment.
    The current VoxelEnv (Discrete(3)) and SurfaceEnv (Discrete(1)) are not
    compatible. Implement a continuous-action environment before enabling TD3.
    """

    algorithm_name = "td3"

    def _build_algorithm(self) -> Any:
        raise NotImplementedError(
            "TD3 requires a continuous (Box) action space environment. "
            "VoxelEnv (Discrete(3)) and SurfaceEnv (Discrete(1)) are not compatible. "
            "Add a Box-action environment and wire it here before using TD3."
        )
