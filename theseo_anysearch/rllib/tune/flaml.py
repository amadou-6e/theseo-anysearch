"""
FLAML tuner — Phase 3.

Uses FLAML's Cost-Frugal Optimiser (CFO) which is aware of trial costs and
allocates the search budget to configurations that give the most improvement
per unit of compute.  Best when trial runtimes vary significantly.

Requires: ``pip install flaml``

YAML key: scheduler: flaml
"""
from __future__ import annotations

from typing import Literal

from theseo_anysearch.rllib.tune.base import BaseTuneConfig
from theseo_anysearch.rllib.tune.models import FLAMLConfig


class FLAMLSearch(BaseTuneConfig):
    """
    Wraps FLAML's ``CFO`` search algorithm with a FIFO scheduler.

    CFO uses randomised local search with a cost model that learns which
    regions of the search space are cheap to evaluate, focusing trials where
    the improvement-per-second is highest.  ``time_budget_s`` caps the total
    wall-clock time for the Tune experiment.
    """

    def __init__(self, cfg: FLAMLConfig, search_space: dict) -> None:
        self._cfg = cfg
        self._search_space = search_space

    def search_space(self) -> dict:
        return self._search_space

    def scheduler(self) -> object:
        from ray.tune.schedulers import FIFOScheduler
        return FIFOScheduler()

    def search_alg(self) -> object:
        try:
            from ray.tune.search.flaml import CFO
        except ImportError as exc:
            raise ImportError(
                "FLAML search requires flaml: pip install flaml"
            ) from exc
        return CFO(
            metric=self._cfg.metric,
            mode=self._cfg.mode,
            time_budget_s=self._cfg.time_budget_s,
        )

    def metric(self) -> str:
        return self._cfg.metric

    def mode(self) -> Literal["min", "max"]:
        return self._cfg.mode
