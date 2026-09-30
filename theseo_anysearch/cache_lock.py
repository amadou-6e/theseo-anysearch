"""Heartbeat-protected locks for content-addressed cache entries."""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_HEARTBEAT_INTERVAL_SECONDS = 5.0


def _read_lock_token(lock_path: Path) -> str | None:
    try:
        return lock_path.read_text(encoding="ascii")
    except OSError:
        return None


def _lock_is_stale(lock_path: Path, timeout_seconds: float) -> bool:
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        # A transient sharing/permission failure is not evidence that an
        # active holder has disappeared. The acquisition loop will retry.
        return False
    return age >= timeout_seconds


def _heartbeat_loop(
    lock_path: Path, token: str, stop: threading.Event, interval: float
) -> None:
    while not stop.wait(interval):
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
    *,
    label: str = "cache",
) -> Iterator[None]:
    """Serialize access to one cache key and reclaim abandoned holders."""

    if timeout_seconds <= 0:
        raise ValueError("cache lock timeout must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.joinpath(f"{cache_key}.lock")
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, token.encode("ascii"))
            finally:
                os.close(descriptor)
            break
        except FileExistsError:
            if _lock_is_stale(lock_path, timeout_seconds):
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {label} cache key {cache_key}"
                ) from None
            time.sleep(0.1)

    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(
            lock_path,
            token,
            stop_heartbeat,
            min(LOCK_HEARTBEAT_INTERVAL_SECONDS, max(0.01, timeout_seconds / 3.0)),
        ),
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
