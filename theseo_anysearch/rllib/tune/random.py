"""
Random Search tuner — Phase 2.

No early stopping; every trial runs to completion.  Useful as a baseline to
benchmark more sophisticated schedulers against.

YAML key: scheduler: random
"""
from __future__ import annotations

from typing import Literal

from theseo_anysearch.rllib.tune.base import BaseTuneConfig
from theseo_anysearch.rllib.tune.models import RandomConfig


class RandomSearch(BaseTuneConfig):
    """
    Wraps Ray Tune's FIFO scheduler (no early stopping) with random sampling.

    All trials run to ``max_t`` iterations regardless of intermediate performance.
    """

    def __init__(self, cfg: RandomConfig, search_space: dict) -> None:
        self._cfg = cfg
        self._search_space = search_space

    def search_space(self) -> dict:
        return self._search_space

    def scheduler(self) -> object:
        from ray.tune.schedulers import FIFOScheduler
        return FIFOScheduler()

    def metric(self) -> str:
        return self._cfg.metric

    def mode(self) -> Literal["min", "max"]:
        return self._cfg.mode

    def search_alg(self) -> None:
        return None
