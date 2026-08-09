"""Composable perception contracts and built-in sensors."""

from theseo_anysearch.environments.perception.base import Perception, PerceptionMetadata, PerceptionParameters
from theseo_anysearch.environments.perception.registry import (
    PerceptionPipeline,
    get_perception,
    perception_names,
    register_perception,
)

# Importing built-ins performs their local, deterministic registrations.
from theseo_anysearch.environments.perception import builtins as _builtins

__all__ = [
    "Perception",
    "PerceptionMetadata",
    "PerceptionParameters",
    "PerceptionPipeline",
    "get_perception",
    "perception_names",
    "register_perception",
]
