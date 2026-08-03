"""Public settings API.

Import settings contracts from their domain modules in new code. The root package
keeps the commonly used API concise and stable.
"""

from pathlib import Path
from typing import Any

from theseo_anysearch.settings.compatibility import _deep_merge
from theseo_anysearch.settings.algorithm import AlgorithmConfig, AlgorithmEnvCompatibilityMixin, ModelConfig
from theseo_anysearch.settings.environment import ActionConfig, ActionExtensionSelector, AgentConfig, EnvConfig, GeometryConfig, HunterAndHuntedConfig, NestedFieldAccessMixin, ObservationConfig, RewardConfig, RewardSelector
from theseo_anysearch.settings.evaluation import EvaluationConfig
from theseo_anysearch.settings.execution import AnyscaleConfig
from theseo_anysearch.settings.root import Settings
from theseo_anysearch.settings.training import TrainingConfig, TrainingEarlyStopConfig


def load_settings(path: Path, overrides: dict | None = None) -> Settings:
    """Load and validate one settings YAML file."""
    from theseo_anysearch.settings.loading import load_settings as _load_settings
    return _load_settings(path, overrides)


def load_experiment(path: Path) -> Any:
    """Load a plain experiment or sweep YAML file."""
    from theseo_anysearch.settings.loading import load_experiment as _load_experiment
    return _load_experiment(path)


__all__ = [
    "ActionConfig", "ActionExtensionSelector", "AgentConfig", "AlgorithmConfig", "AlgorithmEnvCompatibilityMixin",
    "AnyscaleConfig", "EnvConfig", "EvaluationConfig", "GeometryConfig",
    "HunterAndHuntedConfig", "ModelConfig", "NestedFieldAccessMixin", "ObservationConfig", "RewardConfig",
    "RewardSelector", "Settings", "TrainingConfig", "TrainingEarlyStopConfig",
    "_deep_merge", "load_experiment", "load_settings",
]
