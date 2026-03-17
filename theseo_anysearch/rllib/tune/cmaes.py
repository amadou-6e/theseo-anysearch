"""
CMA-ES tuner — Phase 3.

Uses the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) via the
nevergrad library.  Strong on low-dimensional (≤20 hyperparameters) continuous
search spaces where gradient information is unavailable.

Requires: ``pip install nevergrad``

YAML key: scheduler: cmaes
"""
from __future__ import annotations

from typing import Literal

from theseo_anysearch.rllib.tune.base import BaseTuneConfig
from theseo_anysearch.rllib.tune.models import CMAESConfig


class CMAESSearch(BaseTuneConfig):
    """
    Wraps ``NevergradSearch`` with the CMA optimizer and a FIFO scheduler.

    CMA-ES maintains a multivariate Gaussian over the search space and adapts
    its covariance matrix each generation to concentrate on promising regions.
    ``sigma0`` controls the initial step size (exploration radius).
    """

    def __init__(self, cfg: CMAESConfig, search_space: dict) -> None:
        self._cfg = cfg
        self._search_space = search_space

    def search_space(self) -> dict:
        return self._search_space

    def scheduler(self) -> object:
        from ray.tune.schedulers import FIFOScheduler
        return FIFOScheduler()

    def search_alg(self) -> object:
        try:
            import nevergrad as ng
            from ray.tune.search.nevergrad import NevergradSearch
        except ImportError as exc:
            raise ImportError(
                "CMA-ES search requires nevergrad: pip install nevergrad"
            ) from exc
        optimizer = ng.optimizers.ParametrizedCMA(sigma=self._cfg.sigma0)
        return NevergradSearch(
            optimizer=optimizer,
            metric=self._cfg.metric,
            mode=self._cfg.mode,
        )

    def metric(self) -> str:
        return self._cfg.metric

    def mode(self) -> Literal["min", "max"]:
        return self._cfg.mode
