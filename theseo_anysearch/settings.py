from __future__ import annotations

from pathlib import Path

import yaml

from theseo_anysearch.models import (
    AlgorithmConfig,
    AnyscaleConfig,
    EnvConfig,
    ModelConfig,
    Settings,
    TrainingConfig,
)
from theseo_anysearch.rllib.algorithms.models import get_algorithm_config_class
from theseo_anysearch.rllib.models.models import get_model_config_class


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base in-place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_experiment(path: Path):
    """
    Load an experiment YAML and return an ExperimentConfig or SweepConfig.
    Delegates to experiments.loader which handles sweep detection and typed config
    resolution.  Kept here as a convenience shim so callers can do::

        from settings import load_experiment
    """
    from theseo_anysearch.experiments.loader import load_experiment as _load
    return _load(path)


def load_settings(path: Path, overrides: dict | None = None) -> Settings:
    """Load a YAML settings file, apply overrides, and return a validated Settings object."""
    raw: dict = yaml.safe_load(path.read_text())
    if overrides:
        _deep_merge(raw, overrides)

    training_raw = raw.get("training", {})
    runner = training_raw.get("runner", "local")
    model_key = training_raw.get("model", "voxel_encoder")

    algo_cls = get_algorithm_config_class(training_raw["algorithm"])
    model_cls = get_model_config_class(model_key)

    if "anyscale" in raw:
        anyscale_raw = raw["anyscale"]
    elif runner == "local":
        anyscale_raw = {
            "cluster_env": "",
            "compute_config": "",
            "project": "",
        }
    else:
        raise KeyError("anyscale")

    return Settings(
        env=EnvConfig(**raw["env"]),
        training=TrainingConfig(**training_raw),
        anyscale=AnyscaleConfig(**anyscale_raw),
        algorithm_config=algo_cls(**raw.get("algorithm_config", {})),
        model_config=model_cls(**raw.get("model_config", {})),
    )
