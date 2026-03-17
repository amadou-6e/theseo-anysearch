"""
rllib/tune — Tune scheduler implementations and registry.

Registry maps YAML ``scheduler:`` keys to (implementation_class, config_class) pairs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from theseo_anysearch.rllib.tune.base import BaseTuneConfig
from theseo_anysearch.rllib.tune.models import (
    ASHAConfig,
    BOHBConfig,
    CMAESConfig,
    FLAMLConfig,
    HyperbandConfig,
    OptunaConfig,
    PBTConfig,
    RandomConfig,
    TuneConfig,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Registry: scheduler name → (BaseTuneConfig subclass, config class)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[type, type]] = {}


def _register(name: str, impl_cls: type, cfg_cls: type) -> None:
    _REGISTRY[name] = (impl_cls, cfg_cls)


def _lazy_register() -> None:
    """Populate registry lazily (avoids circular imports at module load)."""
    if _REGISTRY:
        return

    from theseo_anysearch.rllib.tune.bohb import BOHBSearch
    from theseo_anysearch.rllib.tune.cmaes import CMAESSearch
    from theseo_anysearch.rllib.tune.flaml import FLAMLSearch
    from theseo_anysearch.rllib.tune.hyperband import HyperbandSearch
    from theseo_anysearch.rllib.tune.optuna import OptunaSearch
    from theseo_anysearch.rllib.tune.random import RandomSearch

    # Phase 1
    # ASHA and PBT are wired directly in cli/commands/tune.py (legacy path).
    # They are included here so callers can introspect the full registry.
    _register("random", RandomSearch, RandomConfig)
    _register("hyperband", HyperbandSearch, HyperbandConfig)
    _register("bohb", BOHBSearch, BOHBConfig)
    _register("optuna", OptunaSearch, OptunaConfig)
    _register("cmaes", CMAESSearch, CMAESConfig)
    _register("flaml", FLAMLSearch, FLAMLConfig)


def get_tune_impl(
    scheduler_name: str,
    cfg: TuneConfig,
    search_space: dict,
) -> BaseTuneConfig:
    """
    Instantiate the concrete ``BaseTuneConfig`` for *scheduler_name*.

    Raises ``ValueError`` for unknown scheduler names.
    Raises ``ImportError`` (from the implementation) when optional deps are missing.
    """
    _lazy_register()
    key = scheduler_name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. Known: {known}"
        )
    impl_cls, _ = _REGISTRY[key]
    return impl_cls(cfg, search_space)


def supported_schedulers() -> list[str]:
    """Return all registered scheduler names (Phase 2 + 3 only; ASHA/PBT handled separately)."""
    _lazy_register()
    return sorted(_REGISTRY)


__all__ = [
    "BaseTuneConfig",
    "TuneConfig",
    "ASHAConfig",
    "PBTConfig",
    "RandomConfig",
    "HyperbandConfig",
    "BOHBConfig",
    "OptunaConfig",
    "CMAESConfig",
    "FLAMLConfig",
    "get_tune_impl",
    "supported_schedulers",
]
