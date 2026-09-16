"""Bounded, hash-verified source-file caching without executing downloaded code."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

MAX_FILE_BYTES = 2 * 1024 * 1024


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_sources(cache: Path, *, base_url: str, hashes: dict[str, str], offline: bool = False,
                   max_file_bytes: int = MAX_FILE_BYTES) -> Path:
    """Publish only complete verified manifests; never repair corrupt caches silently."""
    if not base_url.startswith("https://") or not hashes:
        raise ValueError("source manifest requires HTTPS and nonempty hashes")
    if type(max_file_bytes) is not int or not 0 < max_file_bytes <= 32 * 1024 * 1024:
        raise ValueError("source file limit must be positive and at most 32 MiB")
    for name, digest in hashes.items():
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in {"..", "."} for part in path.parts) or "\\" in name or ":" in name:
            raise ValueError("unsafe source manifest path")
        if not name or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source manifest")
    identity = hashlib.sha256(json.dumps({"base_url": base_url, "hashes": hashes}, sort_keys=True).encode()).hexdigest()
    cache = cache.resolve()
    target = cache / identity

    def verify(root: Path) -> None:
        for name, digest in hashes.items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("source cache is incomplete or escapes its root")
            if path.stat().st_size > max_file_bytes or _digest(path) != digest:
                raise ValueError("source cache hash mismatch")

    if target.exists():
        verify(target)
        return target
    if offline:
        raise FileNotFoundError("offline generation requires a complete verified source cache")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="download-", dir=cache))
    try:
        for name, digest in hashes.items():
            with urlopen(base_url.rstrip("/") + "/" + name, timeout=20) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("source download redirected away from HTTPS")
                data = response.read(max_file_bytes + 1)
            if len(data) > max_file_bytes or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("source download size or hash mismatch")
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        verify(temporary)
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            verify(target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
