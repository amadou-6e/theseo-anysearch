"""
Optuna tuner — Phase 3.

Uses Optuna's Tree-structured Parzen Estimator (TPE) as the search algorithm,
paired with ASHA early stopping.  Best choice for rich continuous search spaces
where the Bayesian model can exploit correlations between hyperparameters.

Requires: ``pip install optuna``

YAML key: scheduler: optuna
"""
from __future__ import annotations

from typing import Literal

from theseo_anysearch.rllib.tune.base import BaseTuneConfig
from theseo_anysearch.rllib.tune.models import OptunaConfig


class OptunaSearch(BaseTuneConfig):
    """
    Pairs ``OptunaSearch`` (TPE) with ``ASHAScheduler`` for early stopping.

    After ``n_startup_trials`` random trials, Optuna builds a probabilistic
    model over the objective and samples from the region most likely to
    improve.  ASHA terminates clearly underperforming trials early.
    """

    def __init__(self, cfg: OptunaConfig, search_space: dict) -> None:
        self._cfg = cfg
        self._search_space = search_space

    def search_space(self) -> dict:
        return self._search_space

    def scheduler(self) -> object:
        from ray.tune.schedulers import ASHAScheduler
        return ASHAScheduler(
            time_attr="training_iteration",
            max_t=self._cfg.max_t,
            grace_period=self._cfg.grace_period,
            reduction_factor=3,
        )

    def search_alg(self) -> object:
        try:
            from ray.tune.search.optuna import OptunaSearch as _OptunaSearch
        except ImportError as exc:
            raise ImportError(
                "Optuna search algorithm requires optuna: pip install optuna"
            ) from exc
        return _OptunaSearch(
            metric=self._cfg.metric,
            mode=self._cfg.mode,
            n_startup_trials=self._cfg.n_startup_trials,
        )

    def metric(self) -> str:
        return self._cfg.metric

    def mode(self) -> Literal["min", "max"]:
        return self._cfg.mode
