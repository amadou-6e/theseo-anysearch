"""Convention-based episode generation-provider discovery and execution."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.imitation.generation_providers import (
    DemonstrationEpisode,
    EpisodeGenerationContext,
    GenerationProvider,
)


class CustomGenerationError(ValueError):
    """Raised when a generation provider violates its explicit contract."""


class GenerationProviderRecord(BaseModel):
    """Validated Python generation function."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    source_path: Path
    source_sha256: str
    generate: GenerationProvider = Field(exclude=True)


def available_python_generation_names(
    source_path: Path | None,
    candidates: tuple[str, ...],
) -> tuple[str, ...]:
    """Return selected names implemented by one Python generation module."""
    if source_path is None:
        return ()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"_theseo_anysearch_generation_probe_{digest[:16]}", source_path
    )
    if spec is None or spec.loader is None:
        raise CustomGenerationError(f"Cannot import custom generation from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    available: list[str] = []
    for name in candidates:
        function = getattr(module, name, None)
        if function is None:
            continue
        if not callable(function):
            raise CustomGenerationError(f"{source_path}: {name} must be callable")
        if len(inspect.signature(function).parameters) != 1:
            raise CustomGenerationError(
                f"{source_path}: {name} must accept exactly one argument"
            )
        available.append(name)
    return tuple(available)


def discover_generation_source(
    config_path: Path | None, provider_name: str | None
) -> Path | None:
    """Discover the conventional sibling ``imitation.py`` module."""
    if config_path is None:
        return None
    if provider_name is None:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        imitation_block = raw.get("imitation") or {}
        selected = (imitation_block.get("generation") or {}).get("provider")
        if not selected:
            return None
    source = config_path.with_name("imitation.py")
    return source if source.is_file() else None


def copy_generation_source(
    config_path: Path | None, destination: Path, provider_name: str | None
) -> Path | None:
    """Archive the exact Python generation source used by a run."""
    source = discover_generation_source(config_path, provider_name)
    if source is None:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination.joinpath("imitation.py")
    shutil.copy2(source, target)
    return target


def load_generation_provider(
    source_path: Path | None, provider_name: str | None
) -> GenerationProviderRecord | None:
    """Load one named function from ``imitation.py`` without fallbacks."""
    if source_path is None or provider_name is None:
        return None
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"_theseo_anysearch_generation_{digest[:16]}", source_path
    )
    if spec is None or spec.loader is None:
        raise CustomGenerationError(f"Cannot import custom generation from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, provider_name, None)
    if not callable(function):
        raise CustomGenerationError(
            f"{source_path} must define callable {provider_name}(context)"
        )
    if len(inspect.signature(function).parameters) != 1:
        raise CustomGenerationError(
            f"{source_path}: {provider_name} must accept exactly one argument"
        )

    def generate(context: EpisodeGenerationContext) -> DemonstrationEpisode:
        return DemonstrationEpisode.model_validate(function(context))

    return GenerationProviderRecord(
        name=provider_name,
        source_path=source_path,
        source_sha256=digest,
        generate=generate,
    )
