"""Trajectory JSON storage: Zstandard on disk, ordinary JSON in memory.

The JSON schema is unchanged. Legacy .json files remain readable; new writes
use .json.zst. Decompression occurs once, never during replay scrubbing.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import zstandard

SUFFIX = ".json.zst"


def read_trajectory(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if path.name.endswith(SUFFIX):
        raw = zstandard.ZstdDecompressor().decompress(raw)
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"trajectory must be a JSON object: {path}")
    return document


def write_trajectory(path: Path, document: dict[str, Any]) -> Path:
    """Atomically publish a complete frame, leaving an old snapshot on failure."""
    if not path.name.endswith(SUFFIX):
        raise ValueError(f"new trajectory paths must end in {SUFFIX}")
    raw = json.dumps(document, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    compressed = zstandard.ZstdCompressor(level=3).compress(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def trajectory_stem(path: Path) -> str:
    return path.name.removesuffix(".zst").removesuffix(".json")


def find_trajectory(directory: Path, stem: str) -> Path:
    """Prefer the new encoding for a logical name, without fallback on corruption."""
    stem = stem.removesuffix(".zst").removesuffix(".json")
    for suffix in (SUFFIX, ".json"):
        candidate = directory / (stem + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"trajectory not found: {directory / stem}")


def list_trajectories(directory: Path, pattern: str = "*") -> list[Path]:
    """List logical trajectories once; metadata and temporary files are excluded."""
    paths = {}
    for suffix in (".json", SUFFIX):
        for path in directory.glob(pattern + suffix):
            stem = trajectory_stem(path)
            if path.is_file() and not stem.endswith("_meta"):
                paths[stem] = path
    return [paths[key] for key in sorted(paths)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/export a saved trajectory as readable JSON.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--step", type=int, help="Zero-based step index; omit for the whole trajectory.")
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten.")
    args = parser.parse_args()
    document = read_trajectory(args.path)
    if args.step is not None:
        steps = document["episode"]["steps"]
        if args.step < 0 or args.step >= len(steps):
            parser.error("step index outside the trajectory")
        document = steps[args.step]
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
