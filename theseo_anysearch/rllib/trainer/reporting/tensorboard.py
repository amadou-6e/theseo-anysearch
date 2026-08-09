"""TensorBoard reporting for project training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from theseo_anysearch.rllib.trainer.results import TrainResult
class _TensorBoardRunWriter:
    """Write per-iteration training metrics to a run-local TensorBoard logdir.

    The writer keeps TensorBoard ownership in the project trainer layer instead of
    relying on RLlib's legacy logger placement, which may write to opaque Ray temp
    locations. Event files are emitted under ``<run_dir>/tensorboard`` so
    ``anysearch tensorboard`` can point at the run directory directly.
    """

    def __init__(self, output_dir: Path) -> None:
        self._log_dir = output_dir.joinpath("tensorboard")
        self._output_dir = output_dir
        self._writer: Any | None = None
        self._disabled_reason: str | None = None

        try:
            from torch.utils.tensorboard import SummaryWriter
        except (ImportError, ModuleNotFoundError) as exc:
            # TensorBoard/torch is genuinely not installed — this is the one
            # case we treat as an expected, benign "optional dependency
            # missing" outcome. Record why, so a user inspecting the run can
            # tell TensorBoard was skipped (and it isn't confused with a run
            # that never intended to use it, or with a real bug below).
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            self._record_disabled_reason()
            return
        # Any other exception here (AttributeError, TypeError, etc. from a
        # code defect in this integration) is a real bug, not a missing
        # dependency — let it propagate instead of masking it as "disabled".

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

    def _record_disabled_reason(self) -> None:
        """Persist why TensorBoard is disabled into the run's artifacts.

        A log warning alone disappears once the process exits; writing a
        small marker file means the degraded/disabled state is still
        discoverable when a user inspects the run directory afterward.
        """
        try:
            from theseo_anysearch.experiments.output import OutputStore

            store = OutputStore(self._output_dir)
            store.write_json(
                "tensorboard_status.json",
                {"enabled": False, "disabled_reason": self._disabled_reason},
            )
        except Exception:
            # Recording the reason is best-effort; never let it block training.
            pass

    @property
    def enabled(self) -> bool:
        """Return True when TensorBoard writing is active."""
        return self._writer is not None

    @property
    def disabled_reason(self) -> str | None:
        """Reason TensorBoard was disabled (only set on genuine ImportError)."""
        return self._disabled_reason

    @property
    def log_dir(self) -> Path:
        """Return the concrete TensorBoard log directory."""
        return self._log_dir

    def log_iteration(self, result: "TrainResult") -> None:
        """Append one iteration of scalar metrics to the event stream."""
        if self._writer is None:
            return

        self._writer.add_scalar(
            "train/episode_reward_mean",
            result.episode_reward_mean,
            result.iteration,
        )
        self._writer.add_scalar(
            "train/episode_len_mean",
            result.episode_len_mean,
            result.iteration,
        )
        self._writer.add_scalar(
            "train/episodes_total",
            result.episodes_total,
            result.iteration,
        )
        self._writer.add_scalar(
            "train/elapsed_s",
            result.elapsed_s,
            result.iteration,
        )
        for tag, value in result.timings.tensorboard_scalars(result.elapsed_s).items():
            self._writer.add_scalar(tag, value, result.iteration)
        self._writer.flush()

    def log_scalars(self, iteration: int, scalars: Mapping[str, float]) -> None:
        """Append additional scalar metrics for one iteration."""
        if self._writer is None:
            return

        for tag, value in scalars.items():
            self._writer.add_scalar(tag, value, iteration)
        self._writer.flush()

    def close(self) -> None:
        """Close the underlying SummaryWriter when it exists."""
        if self._writer is None:
            return
        self._writer.close()
