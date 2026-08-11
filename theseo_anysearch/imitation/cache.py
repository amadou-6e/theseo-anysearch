"""Concurrency-safe content-addressed cache for imitation checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from theseo_anysearch.imitation.models import (
    DemonstrationManifest,
    ImitationConfig,
    ImitationResult,
)

CACHE_SCHEMA_VERSION = 1
_LOCK_HEARTBEAT_INTERVAL_SECONDS = 5.0


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
        "model": {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "structure": str(model),
            "state": state_contract,
        },
        "torch_version": torch.__version__,
        "pretraining": imitation.pretraining.model_dump(mode="json"),
        "handoff": imitation.handoff.model_dump(mode="json"),
    }
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), contract


def _read_lock_token(lock_path: Path) -> str | None:
    try:
        return lock_path.read_text(encoding="ascii")
    except OSError:
        return None


def _lock_is_stale(lock_path: Path, timeout_seconds: float) -> bool:
    """A lock is stale once its heartbeat has gone silent past the timeout.

    Staleness is judged from the lock file's own mtime rather than from how
    long any individual waiter has been polling, so a holder doing
    long-running work (e.g. behavior cloning) is refreshed by its own
    heartbeat and is never mistaken for a crashed process.
    """
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return True
    return age >= timeout_seconds


def _heartbeat_loop(lock_path: Path, token: str, stop: threading.Event) -> None:
    """Refresh a held lock's mtime until released or reclaimed out from under us."""
    while not stop.wait(_LOCK_HEARTBEAT_INTERVAL_SECONDS):
        if _read_lock_token(lock_path) != token:
            return
        try:
            os.utime(lock_path, None)
        except OSError:
            return


@contextmanager
def cache_key_lock(
    cache_dir: Path,
    cache_key: str,
    timeout_seconds: float,
) -> Iterator[None]:
    """Serialize publishers and consumers for one cache key.

    ``timeout_seconds`` bounds both how long a waiter is willing to wait and
    how long a lock may go without a heartbeat before it is reclaimed as
    abandoned. Each holder writes a unique token into the lock file and only
    ever removes the file itself when that token is still present, so a
    holder whose lock was reclaimed while it was still (slowly) working can
    never delete a later holder's active lock.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.joinpath(f"{cache_key}.lock")
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    while not acquired:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, token.encode("ascii"))
            os.close(descriptor)
            acquired = True
        except FileExistsError:
            if _lock_is_stale(lock_path, timeout_seconds):
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for imitation cache key {cache_key}"
                ) from None
            time.sleep(0.1)

    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(lock_path, token, stop_heartbeat),
        daemon=True,
    )
    heartbeat.start()
    try:
        yield
    finally:
        stop_heartbeat.set()
        heartbeat.join()
        if _read_lock_token(lock_path) == token:
            lock_path.unlink(missing_ok=True)


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
