from __future__ import annotations

from theseo_anysearch.rllib.runner.base import Runner
from theseo_anysearch.rllib.trainer.base import Trainer


class LocalRunner(Runner):
    """Runs trainer.train() directly in the current process. No Ray cluster required."""

    def __init__(self) -> None:
        self._status = "idle"

    def submit(self, trainer: Trainer) -> dict:
        self._status = "running"
        try:
            result = trainer.train()
            self._status = "done"
            return result
        except Exception:
            self._status = "error"
            raise

    def status(self) -> str:
        return self._status

    def cancel(self) -> None:
        self._status = "cancelled"
