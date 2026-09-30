"""Lifecycle execution for one RLlib training run."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.rllib.trainer.runtime import (
    _append_trainer_stage_log,
    _log_trainer_stage,
)


class IterationExecution(BaseModel):
    """Raw output and duration of one algorithm iteration.

    Parameters
    ----------
    result : dict[str, Any]
        Raw mapping returned by RLlib.
    duration_s : float
        Wall-clock duration of the iteration in seconds.
    """

    model_config = ConfigDict(extra="forbid")

    result: dict[str, Any] = Field(default_factory=dict)
    duration_s: float


class TrainingLifecycle:
    """Build and execute an RLlib algorithm with lifecycle logging.

    Parameters
    ----------
    output_dir : pathlib.Path
        Training run output directory.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def ensure_algorithm(
        self,
        algorithm: Any,
        builder: Callable[[], Any],
    ) -> Any:
        """Return an existing algorithm or construct it once.

        Parameters
        ----------
        algorithm : Any
            Existing algorithm instance, when already built.
        builder : Callable[[], Any]
            Algorithm construction callback.

        Returns
        -------
        Any
            Ready algorithm instance.
        """
        if algorithm is not None:
            return algorithm
        _log_trainer_stage("Building algorithm instance")
        _append_trainer_stage_log(
            self._output_dir,
            "Building algorithm instance",
        )
        algorithm = builder()
        _log_trainer_stage("Algorithm instance ready")
        _append_trainer_stage_log(
            self._output_dir,
            "Algorithm instance ready",
        )
        return algorithm

    def run_iteration(
        self,
        algorithm: Any,
        iteration: int,
        total_iterations: int,
    ) -> IterationExecution:
        """Execute and time one algorithm training iteration.

        Parameters
        ----------
        algorithm : Any
            RLlib algorithm or compatible object exposing train.
        iteration : int
            One-based iteration number.
        total_iterations : int
            Configured total iteration count.

        Returns
        -------
        IterationExecution
            Raw RLlib result and measured duration.
        """
        message = f"Starting train iteration {iteration}/{total_iterations}"
        _log_trainer_stage(message)
        _append_trainer_stage_log(self._output_dir, message)
        started = time.perf_counter()
        result = algorithm.train()
        duration_s = time.perf_counter() - started
        message = (
            f"Finished train iteration {iteration}/{total_iterations} "
            f"in {duration_s:.2f}s"
        )
        _log_trainer_stage(message)
        _append_trainer_stage_log(self._output_dir, message)
        return IterationExecution(result=result, duration_s=duration_s)
