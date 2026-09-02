"""Node-local staging and configuration for compiled-world residency."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from theseo_anysearch.cache_lock import cache_key_lock
from theseo_anysearch.worlds.compiler import (
    CompiledWorld,
    WorldPackCorruptError,
    validate_compiled_world,
)


@dataclass(frozen=True)
class WorldResidencySettings:
    """Per-process decoded-cache and prefetch limits."""

    maximum_decoded_bytes: int = 256 * 1024 * 1024
    prefetch_margin: int = 2
    lock_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.maximum_decoded_bytes <= 0:
            raise ValueError("maximum_decoded_bytes must be positive")
        if self.prefetch_margin < 0:
            raise ValueError("prefetch_margin cannot be negative")
        if self.lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")


def stage_compiled_world(
    world: CompiledWorld,
    node_cache: Path,
    *,
    lock_timeout_seconds: float = 300.0,
) -> CompiledWorld:
    """Atomically stage one validated pack in a node-local content cache."""

    identity = world.manifest.identity_sha256
    node_cache.mkdir(parents=True, exist_ok=True)
    destination = node_cache.joinpath(identity)
    with cache_key_lock(
        node_cache,
        identity,
        lock_timeout_seconds,
        label="node-local world pack",
    ):
        if destination.is_dir():
            try:
                return validate_compiled_world(destination)
            except WorldPackCorruptError:
                quarantine = node_cache.joinpath(
                    f".{identity}.{uuid.uuid4().hex}.corrupt"
                )
                os.replace(destination, quarantine)
                shutil.rmtree(quarantine)
        temporary = node_cache.joinpath(f".{identity}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir()
        try:
            for source in world.root.iterdir():
                if source.is_file():
                    shutil.copy2(source, temporary.joinpath(source.name))
            staged = validate_compiled_world(temporary)
            temporary.joinpath("staging.json").write_text(
                json.dumps(
                    {
                        "identity_sha256": identity,
                        "source": str(world.root.resolve()),
                        "process_id": os.getpid(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, destination)
            return CompiledWorld(root=destination, manifest=staged.manifest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def resolve_worker_world(source: Path, node_cache: Path | None = None) -> CompiledWorld:
    """Validate a shared pack and optionally stage it into a worker-local cache."""

    world = validate_compiled_world(source.resolve())
    if node_cache is None:
        return world
    return stage_compiled_world(world, node_cache.resolve())
