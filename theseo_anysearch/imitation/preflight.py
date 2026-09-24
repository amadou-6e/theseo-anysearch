"""Resolve imitation providers and reject invalid YAML before Ray starts."""

from __future__ import annotations

from pathlib import Path

from theseo_anysearch.experiments.custom_imitation import (
    available_python_generation_names,
    discover_generation_source,
)
from theseo_anysearch.imitation.generation_providers import BUILT_IN_GENERATION_PROVIDERS
from theseo_anysearch.imitation.models import ImitationConfig
from theseo_anysearch.imitation.sampling_providers import BUILT_IN_SAMPLING_PROVIDERS


class ImitationPreflightError(ValueError):
    """Raised when a configured imitation provider cannot be resolved."""


def preflight_imitation_providers(
    imitation: ImitationConfig,
    config_path: Path | None = None,
) -> None:
    """Resolve the selected generation/sampling provider names or fail loudly."""
    if not imitation.enabled:
        return

    generation_name = imitation.generation.provider.name
    source = discover_generation_source(config_path, generation_name)
    python_names = available_python_generation_names(source, (generation_name,))
    is_built_in = generation_name in BUILT_IN_GENERATION_PROVIDERS

    if is_built_in and generation_name in python_names:
        raise ImitationPreflightError(
            f"imitation.generation.provider: {generation_name!r} is a reserved "
            "built-in name and cannot be redefined in imitation.py"
        )
    if not is_built_in and generation_name not in python_names:
        raise ImitationPreflightError(
            f"imitation.generation.provider: unknown generation provider "
            f"{generation_name!r}; built-in names: "
            f"{sorted(BUILT_IN_GENERATION_PROVIDERS)}"
        )

    sampling_name = imitation.sampling.provider.name
    if sampling_name not in BUILT_IN_SAMPLING_PROVIDERS:
        raise ImitationPreflightError(
            f"imitation.sampling.provider: unknown sampling provider "
            f"{sampling_name!r}; built-in names: {sorted(BUILT_IN_SAMPLING_PROVIDERS)}"
        )
