"""
BOHB tuner — Phase 2.

Combines Hyperband early-stopping with Bayesian optimisation via the
Tree-structured Parzen Estimator (TPE).  State-of-the-art for small-to-medium
compute budgets where both exploration and exploitation matter.

Requires: ``pip install "ray[tune]" ConfigSpace``

YAML key: scheduler: bohb
"""
from __future__ import annotations

from typing import Literal

from theseo_anysearch.rllib.tune.base import BaseTuneConfig
from theseo_anysearch.rllib.tune.models import BOHBConfig


class BOHBSearch(BaseTuneConfig):
    """
    Pairs ``HyperBandForBOHB`` scheduler with ``TuneBOHB`` search algorithm.

    ``TuneBOHB`` builds a Bayesian model over completed rung results and
    proposes configs that maximise the acquisition function, while
    ``HyperBandForBOHB`` stops underperforming trials early.
    """

    def __init__(self, cfg: BOHBConfig, search_space: dict) -> None:
        self._cfg = cfg
        self._search_space = search_space

    def search_space(self) -> dict:
        return self._search_space

    def scheduler(self) -> object:
        try:
            from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
        except ImportError as exc:
            raise ImportError(
                "BOHB scheduler requires ray[tune] and ConfigSpace: "
                "pip install \"ray[tune]\" ConfigSpace"
            ) from exc
        return HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=self._cfg.max_t,
            reduction_factor=self._cfg.reduction_factor,
        )

    def search_alg(self) -> object:
        try:
            from ray.tune.search.bohb import TuneBOHB
        except ImportError as exc:
            raise ImportError(
                "BOHB search algorithm requires ConfigSpace: "
                "pip install ConfigSpace"
            ) from exc
        return TuneBOHB(metric=self._cfg.metric, mode=self._cfg.mode)

    def metric(self) -> str:
        return self._cfg.metric

    def mode(self) -> Literal["min", "max"]:
        return self._cfg.mode
