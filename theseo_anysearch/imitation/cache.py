"""Concurrency-safe content-addressed cache for imitation checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from theseo_anysearch.cache_lock import cache_key_lock

from theseo_anysearch.imitation.models import (
    DemonstrationManifest,
    ImitationConfig,
    ImitationResult,
)

CACHE_SCHEMA_VERSION = 2


def pretraining_cache_key(
    model: Any,
    dataset_manifest: DemonstrationManifest,
    imitation: ImitationConfig,
    *,
    policy_id: str,
) -> tuple[str, dict[str, Any]]:
    """Return a stable key and human-readable compatibility contract."""

    state_contract = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(model.state_dict().items())
    }
    contract = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "policy_id": policy_id,
        "dataset_fingerprint": dataset_manifest.fingerprint,
        "world": {
            "schema_version": dataset_manifest.world_schema_version,
            "coordinate_type": dataset_manifest.coordinate_type,
            "coordinate_convention": dataset_manifest.coordinate_convention,
            "storage_coordinate_convention": (
                dataset_manifest.storage_coordinate_convention
            ),
            "source_origin": list(dataset_manifest.source_origin),
            "extent": list(dataset_manifest.world_extent),
            "identity_sha256": dataset_manifest.world_identity_sha256,
        },
        "model": {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "structure": str(model),
            "state": state_contract,
        },
        "torch_version": torch.__version__,
        "pretraining": imitation.pretraining.model_dump(mode="json"),
        "handoff": imitation.handoff.model_dump(mode="json"),
        "sampling": imitation.sampling.provider.model_dump(mode="json"),
    }
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), contract


def load_cached_pretraining(
    model: Any,
    cache_dir: Path,
    cache_key: str,
    contract: dict[str, Any],
    output_dir: Path,
) -> ImitationResult | None:
    """Load and materialize a compatible cached checkpoint."""

    entry = cache_dir.joinpath(cache_key)
    manifest_path = entry.joinpath("manifest.json")
    checkpoint_path = entry.joinpath("policy_state.pt")
    result_path = entry.joinpath("result.json")
    if not (manifest_path.is_file() and checkpoint_path.is_file() and result_path.is_file()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != contract or not manifest.get("complete", False):
        return None
    payload = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
    model.load_state_dict(payload["model_state"])
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = output_dir.joinpath("policy_state.pt")
    shutil.copy2(checkpoint_path, materialized)
    result = ImitationResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    return result.model_copy(
        update={
            "checkpoint_path": str(materialized),
            "cache_hit": True,
            "cache_key": cache_key,
        }
    )


def publish_cached_pretraining(
    cache_dir: Path,
    cache_key: str,
    contract: dict[str, Any],
    result: ImitationResult,
) -> None:
    """Atomically publish a completed cache entry."""

    entry = cache_dir.joinpath(cache_key)
    if entry.is_dir():
        return
    temporary = cache_dir.joinpath(f".{cache_key}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    shutil.copy2(result.checkpoint_path, temporary.joinpath("policy_state.pt"))
    temporary.joinpath("result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    temporary.joinpath("manifest.json").write_text(
        json.dumps({"complete": True, "contract": contract}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, entry)
