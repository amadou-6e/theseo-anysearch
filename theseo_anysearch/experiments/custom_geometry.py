"""Convention-discovered, validated Python geometry providers."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.settings.environment.geometry import GeometrySource


class CustomGeometryError(ValueError):
    """Raised when a geometry provider violates its contract."""


@runtime_checkable
class GeometryWorld(Protocol):
    """Read-only bounded occupancy view exposed to generators."""

    extent: tuple[int, int, int]

    def occupied(self, coordinate: tuple[int, int, int]) -> bool: ...

    def occupied_in_region(
        self,
        minimum: tuple[int, int, int],
        maximum_exclusive: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], ...]: ...


class GeometryTaskRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start: tuple[int, int, int] | None = None
    goals: tuple[tuple[int, int, int], ...] = ()
    max_steps: int
    action_mode: str


class GeometryContext(BaseModel):
    """Immutable inputs supplied to a geometry provider."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    seed: int
    attempt: int
    extent: tuple[int, int, int]
    task: GeometryTaskRequirements
    parameters: dict[str, Any] = Field(default_factory=dict)
    world: GeometryWorld = Field(exclude=True)


class GeometryProposal(BaseModel):
    """Inert proposed sources; environments validate before installation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    proposal_id: str
    version: str = "1"
    sources: tuple[GeometrySource, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


GeometryFunction = Callable[[GeometryContext], GeometryProposal]


class GeometryProvider(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    source_path: Path
    source_sha256: str
    generate: GeometryFunction = Field(exclude=True)
    native_abi: int | None = None


class RuntimeGeometryWorld:
    """Read-only adapter around the native bounded world-query API."""

    def __init__(self, world: Any, extent: tuple[int, int, int]) -> None:
        self._occupied = world.world_occupied
        self._occupied_in_region = world.world_occupied_in_region
        self.extent = extent

    def occupied(self, coordinate: tuple[int, int, int]) -> bool:
        return bool(self._occupied(coordinate))

    def occupied_in_region(
        self,
        minimum: tuple[int, int, int],
        maximum_exclusive: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            tuple(item)
            for item in self._occupied_in_region(
                minimum, maximum_exclusive, 100_000
            )
        )


def discover_geometry_source(
    config_path: Path | None, provider_name: str | None
) -> Path | None:
    """Discover the conventional sibling ``geometry.py`` module."""
    if config_path is None:
        return None
    if provider_name is None:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        owners = [raw.get("env") or {}, raw.get("evaluation") or {}]
        for stage in (raw.get("staging") or {}).get("stages") or []:
            owners.extend((stage.get("env") or {}, stage.get("evaluation") or {}))
        if not any(((owner.get("geometry") or {}).get("provider")) for owner in owners):
            return None
    source = config_path.with_name("geometry.py")
    return source if source.is_file() else None


def copy_geometry_source(
    config_path: Path | None, destination: Path, provider_name: str | None
) -> Path | None:
    source = discover_geometry_source(config_path, provider_name)
    if source is None:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination.joinpath("geometry.py")
    shutil.copy2(source, target)
    return target


def load_geometry_provider(
    source_path: Path | None, provider_name: str | None
) -> GeometryProvider | None:
    if source_path is None or provider_name is None:
        return None
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"_theseo_anysearch_geometry_{digest[:16]}", source_path
    )
    if spec is None or spec.loader is None:
        raise CustomGeometryError(f"Cannot import geometry provider from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, provider_name, None)
    if not callable(function):
        raise CustomGeometryError(
            f"{source_path} must define callable {provider_name}(context)"
        )
    if len(inspect.signature(function).parameters) != 1:
        raise CustomGeometryError(
            f"{source_path}: {provider_name} must accept exactly one argument"
        )

    def generate(context: GeometryContext) -> GeometryProposal:
        try:
            return GeometryProposal.model_validate(function(context))
        except Exception as exc:
            raise CustomGeometryError(
                f"geometry provider {provider_name!r} returned an invalid proposal: {exc}"
            ) from exc

    return GeometryProvider(
        name=provider_name,
        source_path=source_path,
        source_sha256=digest,
        generate=generate,
    )


def load_native_geometry_provider(
    library_path: Path, provider_name: str
) -> GeometryProvider:
    """Describe a native provider; invocation stays in Rust to scope callbacks."""
    if not provider_name.isidentifier():
        raise CustomGeometryError(f"invalid native geometry provider name {provider_name!r}")
    return GeometryProvider(
        name=provider_name,
        source_path=library_path,
        source_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),
        generate=lambda context: (_ for _ in ()).throw(
            CustomGeometryError("native geometry providers must be invoked by the environment")
        ),
        native_abi=1,
    )


def proposal_identity(provider: GeometryProvider, context: GeometryContext, proposal: GeometryProposal) -> str:
    payload = {
        "provider": provider.name,
        "source_sha256": provider.source_sha256,
        "parameters": context.parameters,
        "proposal": proposal.model_dump(mode="json"),
    }
    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _EmptyWorld:
    def __init__(self, extent: tuple[int, int, int]) -> None:
        self.extent = extent

    def occupied(self, coordinate: tuple[int, int, int]) -> bool:
        return False

    def occupied_in_region(self, minimum, maximum_exclusive):
        return ()


def preflight_geometry_provider(geometry: Any, env: Any, config_path: Path | None) -> GeometryProvider | None:
    """Validate discovery, signature, output shape, and fixed-seed determinism."""
    selector = geometry.provider
    if selector is None:
        return None
    extent = tuple(geometry.extent or (geometry.grid_size,) * 3)
    max_steps = int(env.max_steps)
    action_mode = str(env.action.mode)
    source = discover_geometry_source(config_path, selector.name)
    if source is None:
        from theseo_anysearch.experiments.native_extensions import (
            NativeExtensionManifest,
            discover_native_manifest,
        )

        manifest_path = discover_native_manifest(config_path)
        if manifest_path is not None:
            manifest = NativeExtensionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if selector.name in manifest.geometries:
                provider = load_native_geometry_provider(
                    manifest_path.parent.joinpath(manifest.library).resolve(),
                    selector.name,
                )
                _preflight_native_provider(
                    provider,
                    extent,
                    seed=int(env.seed),
                    max_steps=max_steps,
                    action_mode=action_mode,
                    parameters=selector.parameters,
                )
                return provider
        expected = config_path.with_name("geometry.py") if config_path else Path("geometry.py")
        raise CustomGeometryError(
            f"geometry provider {selector.name!r} requires {expected} or a compiled native export"
        )
    provider = load_geometry_provider(source, selector.name)
    assert provider is not None
    context = GeometryContext(
        seed=int(env.seed),
        attempt=1,
        extent=extent,
        task=GeometryTaskRequirements(max_steps=max_steps, action_mode=action_mode),
        parameters=selector.parameters,
        world=_EmptyWorld(extent),
    )
    first = provider.generate(context)
    second = provider.generate(context)
    if first != second:
        raise CustomGeometryError(
            f"geometry provider {selector.name!r} is nondeterministic for a fixed context"
        )
    return provider


def _preflight_native_provider(
    provider: GeometryProvider,
    extent: tuple[int, int, int],
    *,
    seed: int,
    max_steps: int,
    action_mode: str,
    parameters: dict[str, Any],
) -> None:
    """Trigger the native ABI's own determinism check before Ray starts.

    ``generate_native_geometry_v1`` (the Rust binding) already invokes a
    native provider twice per call and rejects a mismatched output as a
    ``ValueError`` -- see ``NativeGeometryV1::invoke_deterministic``. That
    guards every real reset, but the first reset can already be well into a
    Ray Tune sweep. This calls it once against a throwaway, obstacle-free
    probe environment so a nondeterministic native provider fails at
    preflight, the same point a nondeterministic Python provider already
    fails at above.
    """
    import json

    from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

    probe = VoxelEnv({"extent": extent, "max_steps": max_steps, "action_mode": action_mode})
    try:
        parameters_json = json.dumps(parameters, sort_keys=True)
        task_json = GeometryTaskRequirements(
            max_steps=max_steps, action_mode=action_mode
        ).model_dump_json()
        try:
            probe._rust_env.generate_native_geometry_v1(
                str(provider.source_path),
                provider.name,
                seed,
                1,
                parameters_json,
                task_json,
            )
        except ValueError as exc:
            raise CustomGeometryError(str(exc)) from exc
    finally:
        probe.close()
