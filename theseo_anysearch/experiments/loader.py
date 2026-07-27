"""Load experiment YAML files and resolve typed configuration blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from theseo_anysearch.experiments.models import ExperimentConfig, SweepConfig
from theseo_anysearch.models import (
    AlgorithmConfig,
    ModelConfig,
)
from theseo_anysearch.rllib.algorithms.models import get_algorithm_config_class
from theseo_anysearch.rllib.models.models import get_model_config_class
from theseo_anysearch.settings import _deep_merge


def _resolve_typed_configs(raw: dict) -> dict:
    """
    Replace algorithm_config and model_config dicts in ``raw`` with their
    typed Pydantic instances, based on training.algorithm / training.model.
    Returns a new dict suitable for ExperimentConfig(**...).
    """
    training_raw = raw.get("training", {})
    algo_key = training_raw.get("algorithm", "")
    model_key = training_raw.get("model", "")

    algo_cls = get_algorithm_config_class(algo_key)
    # Experiment YAMLs don't require training.model; default to VoxelEncoderConfig
    # so that use_position_encoding / encoder_depth are accepted.
    model_cls = get_model_config_class(model_key or "voxel_encoder")

    out = dict(raw)
    out["algorithm_config"] = algo_cls(**raw.get("algorithm_config", {}))
    out["model_config"] = model_cls(**raw.get("model_config", {}))
    return out


def load_experiment(path: Path) -> Union[ExperimentConfig, SweepConfig]:
    """
    Load an experiment YAML and return either an ExperimentConfig or SweepConfig.

    - If the YAML has a top-level ``sweep:`` key → SweepConfig
    - Otherwise → ExperimentConfig

    Call ``expand_sweep(sweep)`` to get a list[ExperimentConfig] from a SweepConfig.

    output_dir resolution:
    - If specified in YAML and relative → resolved relative to the YAML's parent directory
    - If absent (using model default)   → set to the YAML's parent directory
    - If absolute                       → used unchanged
    """
    raw: dict = yaml.safe_load(path.read_text()) or {}
    yaml_dir = path.resolve().parent

    if "sweep" in raw:
        sweep_raw = dict(raw["sweep"])
        sweep_raw["description"] = raw.get("description", "")
        return SweepConfig(**sweep_raw)

    resolved = _resolve_typed_configs(raw)
    config = ExperimentConfig(**resolved)

    # Resolve output_dir relative to the YAML's parent
    has_output_dir = "output_dir" in raw.get("experiment", {})
    out = config.experiment.output_dir
    if has_output_dir and not out.is_absolute():
        abs_out = (yaml_dir / out).resolve()
    elif not has_output_dir:
        abs_out = yaml_dir
    else:
        abs_out = out  # already absolute

    return config.model_copy(
        update={"experiment": config.experiment.model_copy(update={"output_dir": abs_out})}
    )


def expand_sweep(sweep: SweepConfig) -> list[ExperimentConfig]:
    """
    Expand a SweepConfig into one ExperimentConfig per sweep entry.
    Each entry deep-merges its keys over the ``base`` config.
    """
    experiments: list[ExperimentConfig] = []
    for entry in sweep.experiments:
        merged = {}
        _deep_merge(merged, sweep.base)
        name = entry.get("name", "")
        entry_without_name = {k: v for k, v in entry.items() if k != "name"}
        _deep_merge(merged, entry_without_name)
        # Inject name into experiment section
        merged.setdefault("experiment", {})["name"] = name
        resolved = _resolve_typed_configs(merged)
        experiments.append(ExperimentConfig(**resolved))
    return experiments
