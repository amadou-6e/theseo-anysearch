"""Trajectory artifact reporting for evaluation episodes."""

from __future__ import annotations

from typing import Any


class TrajectoryReporter:
    """Record periodic and best evaluation trajectories.

    Parameters
    ----------
    writer : Any
        Concrete single-agent or multi-agent trajectory writer.
    """

    def __init__(self, writer: Any) -> None:
        self._writer = writer

    @classmethod
    def create(
        cls,
        output_store: Any,
        *,
        trajectory_every: int,
        best_trajectory: bool,
        multi_agent: bool,
        enabled: bool,
    ) -> "TrajectoryReporter | None":
        """Create a trajectory reporter when recording is required.

        Parameters
        ----------
        output_store : Any
            Training artifact output store.
        trajectory_every : int
            Periodic trajectory interval.
        best_trajectory : bool
            Whether to retain the best trajectory.
        multi_agent : bool
            Whether episodes use the multi-agent trajectory schema.
        enabled : bool
            Additional reason to collect trajectories, such as early stopping.

        Returns
        -------
        TrajectoryReporter | None
            Configured reporter, or None when recording is disabled.
        """
        if not (trajectory_every or best_trajectory or enabled):
            return None
        from theseo_anysearch.experiments.trajectory import (
            MultiTrajectoryWriter,
            TrajectoryWriter,
        )

        writer_type = MultiTrajectoryWriter if multi_agent else TrajectoryWriter
        writer = writer_type(
            output_store,
            trajectory_every,
            best_trajectory,
        )
        return cls(writer)

    def record(
        self,
        episodes: list[Any],
        *,
        iteration: int,
        reward_mean: float,
        experiment_name: str,
        run_id: str,
        force: bool,
    ) -> bool:
        """Record evaluated episodes and finalize iteration artifacts.

        Parameters
        ----------
        episodes : list[Any]
            Completed evaluation episodes.
        iteration : int
            Current training iteration.
        reward_mean : float
            Mean evaluation reward used to rank best trajectories.
        experiment_name : str
            Experiment identifier written to trajectory metadata.
        run_id : str
            Run identifier written to trajectory metadata.
        force : bool
            Whether to write periodic output regardless of interval.

        Returns
        -------
        bool
            Whether a new best trajectory was written.
        """
        for episode in episodes:
            self._writer.record(episode)
        written = self._writer.on_iteration_end(
            iteration,
            reward_mean,
            experiment_name,
            run_id,
            force=force,
        )
        return "trajectories/best.json" in written
