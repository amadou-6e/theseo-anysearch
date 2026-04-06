from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class OutputStore:
    """
    Read/write experiment artifacts under a run directory.

    All paths are relative to ``run_dir``.  Currently backed by the local
    filesystem; switching to cloud (s3://, gs://, az://) requires replacing
    the internals with fsspec calls while keeping the same interface.
    """

    def __init__(self, run_dir: Path) -> None:
        self._root = run_dir
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def write_json(self, rel_path: str, data: Any) -> Path:
        dest = self._root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2, default=str))
        return dest

    def read_json(self, rel_path: str) -> Any:
        return json.loads((self._root / rel_path).read_text())

    def exists(self, rel_path: str) -> bool:
        return (self._root / rel_path).exists()

    # ------------------------------------------------------------------
    # Bytes helpers (for protobuf, etc.)
    # ------------------------------------------------------------------

    def write_bytes(self, rel_path: str, data: bytes) -> Path:
        dest = self._root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def read_bytes(self, rel_path: str) -> bytes:
        return (self._root / rel_path).read_bytes()

    # ------------------------------------------------------------------
    # Directory listing
    # ------------------------------------------------------------------

    def list(self, prefix: str = "") -> list[str]:
        """Return relative paths of all files under ``prefix``."""
        base = self._root / prefix if prefix else self._root
        if not base.exists():
            return []
        return [
            str(p.relative_to(self._root)).replace("\\", "/")
            for p in base.rglob("*")
            if p.is_file()
        ]

    def list_dirs(self, prefix: str = "") -> list[str]:
        """Return relative paths of direct child directories under ``prefix``."""
        base = self._root / prefix if prefix else self._root
        if not base.exists():
            return []
        return [
            str(p.relative_to(self._root)).replace("\\", "/")
            for p in sorted(base.iterdir())
            if p.is_dir()
        ]

    # ------------------------------------------------------------------
    # YAML snapshot
    # ------------------------------------------------------------------

    def write_yaml(self, rel_path: str, source_path: Path) -> Path:
        """Copy the source YAML file into the run directory as-is."""
        dest = self._root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        return dest
