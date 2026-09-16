"""
Hyperband tuner — Phase 2.

Synchronous variant of ASHA.  All trials in a bracket are evaluated together
before promotion decisions are made, which produces more stable results than
ASHA but requires all trials to run to at least ``grace_period`` before any
are stopped.

YAML key: scheduler: hyperband
"""
from __future__ import annotations

from typing import Literal

from theseo_anysearch.rllib.tune.base import BaseTuneConfig
from theseo_anysearch.rllib.tune.models import HyperbandConfig


class HyperbandSearch(BaseTuneConfig):
    """
    Wraps Ray Tune's ``HyperBandScheduler``.

    Uses synchronous successive halving: each rung collects all surviving
    trials before eliminating the bottom ``1/reduction_factor`` fraction.
    """

    def __init__(self, cfg: HyperbandConfig, search_space: dict) -> None:
        self._cfg = cfg
        self._search_space = search_space

    def search_space(self) -> dict:
        return self._search_space

    def scheduler(self) -> object:
        from ray.tune.schedulers import HyperBandScheduler
        return HyperBandScheduler(
            time_attr="training_iteration",
            max_t=self._cfg.max_t,
            reduction_factor=self._cfg.reduction_factor,
        )

    def metric(self) -> str:
        return self._cfg.metric

    def mode(self) -> Literal["min", "max"]:
        return self._cfg.mode

    def search_alg(self) -> None:
        return None
