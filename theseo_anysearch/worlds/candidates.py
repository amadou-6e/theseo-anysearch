"""Compiled, bounded, deterministic candidate-index access."""

from __future__ import annotations

import hashlib
import json
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_DATA_FILE = "candidates.bin"
CANDIDATE_INDEX_FILE = "candidates.idx"
_RECORD = struct.Struct("<IIIBfIII")

CandidateKind = Literal["spawn", "goal", "surface", "portal"]
_KIND_ID: dict[CandidateKind, int] = {
    "spawn": 0,
    "goal": 1,
    "surface": 2,
    "portal": 3,
}
_ID_KIND = {value: key for key, value in _KIND_ID.items()}


class CandidateRecord(BaseModel):
    """One public-coordinate candidate returned without exposing storage chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: tuple[int, int, int]
    kind: CandidateKind
    quality: float = Field(ge=0.0, le=1.0)
    region: tuple[int, int, int]


@dataclass(frozen=True)
class CandidateQueryBudget:
    """Host-owned counters shared by all candidate queries in one reset."""

    maximum_queries: int = 64
    maximum_results: int = 4096


class CandidateBudgetExceeded(RuntimeError):
    """A provider exceeded its configured reset query budget."""


class CandidateIndexHandle:
    """Lazy reader for one immutable, content-addressed candidate index."""

    def __init__(
        self,
        root: Path,
        *,
        world_identity: str | None = None,
        budget: CandidateQueryBudget | None = None,
    ) -> None:
        self._root = root
        self._world_identity = world_identity
        self._budget = budget or CandidateQueryBudget()
        self._queries = 0
        self._results = 0
        payload = json.loads(root.joinpath(CANDIDATE_INDEX_FILE).read_text("utf-8"))
        if payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("unsupported candidate index schema")
        stored_identity = payload.get("world_identity_sha256")
        if world_identity is not None and stored_identity != world_identity:
            raise ValueError("candidate index belongs to a different world")
        if not isinstance(stored_identity, str) or len(stored_identity) != 64:
            raise ValueError("candidate index world identity is invalid")
        self._world_identity = stored_identity
        if payload.get("record_size") != _RECORD.size:
            raise ValueError("candidate index record size is invalid")
        data = root.joinpath(CANDIDATE_DATA_FILE)
        if _sha256_file(data) != payload.get("data_sha256"):
            raise ValueError("candidate data checksum mismatch")
        self._entries = tuple(payload.get("entries", ()))
        data_size = data.stat().st_size
        previous_end = 0
        for entry in self._entries:
            offset = entry.get("offset")
            length = entry.get("byte_length")
            count = entry.get("count")
            if (
                not isinstance(offset, int)
                or not isinstance(length, int)
                or not isinstance(count, int)
                or offset != previous_end
                or length != count * _RECORD.size
                or offset + length > data_size
                or entry.get("kind") not in _KIND_ID
                or not isinstance(entry.get("region"), list)
                or len(entry["region"]) != 3
            ):
                raise ValueError("candidate index entry is invalid")
            previous_end = offset + length
        if previous_end != data_size:
            raise ValueError("candidate index does not cover candidate data")
        self._cache: dict[
            tuple[str, tuple[int, int, int]], tuple[CandidateRecord, ...]
        ] = {}
        self._latency_ns = 0
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def world_identity(self) -> str:
        return self._world_identity

    def sample(
        self,
        count: int,
        kind: CandidateKind,
        *,
        seed: int,
        stream: int,
        near: tuple[int, int, int] | None = None,
        radius: int | None = None,
        region: tuple[int, int, int] | None = None,
        minimum_quality: float = 0.0,
    ) -> tuple[CandidateRecord, ...]:
        """Return a cache-order-independent bounded deterministic sample."""

        if count < 0 or radius is not None and radius < 0:
            raise ValueError("candidate count and radius must be non-negative")
        if not 0 <= seed <= 2**64 - 1 or not 0 <= stream <= 2**64 - 1:
            raise ValueError("candidate seed and stream must fit u64")
        if (near is None) != (radius is None):
            raise ValueError("near and radius must be supplied together")
        self._queries += 1
        if self._queries > self._budget.maximum_queries:
            raise CandidateBudgetExceeded("candidate query budget exhausted")
        remaining = self._budget.maximum_results - self._results
        if count > remaining:
            raise CandidateBudgetExceeded("candidate result budget exhausted")
        started = time.perf_counter_ns()
        candidates = self._read(kind, region)
        if near is not None and radius is not None:
            radius_squared = radius * radius
            candidates = [
                item
                for item in candidates
                if sum((item.position[axis] - near[axis]) ** 2 for axis in range(3))
                <= radius_squared
            ]
        candidates = [item for item in candidates if item.quality >= minimum_quality]
        candidates.sort(key=lambda item: _rank(seed, stream, item))
        result = tuple(candidates[:count])
        self._results += len(result)
        self._latency_ns += time.perf_counter_ns() - started
        return result

    @property
    def statistics(self) -> dict[str, int]:
        """Return reset-local query latency and decoded-bucket cache effects."""

        return {
            "queries": self._queries,
            "results": self._results,
            "latency_ns": self._latency_ns,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    def _read(
        self, kind: CandidateKind, region: tuple[int, int, int] | None
    ) -> list[CandidateRecord]:
        selected = [
            entry
            for entry in self._entries
            if entry["kind"] == kind
            and (region is None or tuple(entry["region"]) == region)
        ]
        records: list[CandidateRecord] = []
        with self._root.joinpath(CANDIDATE_DATA_FILE).open("rb") as stream:
            for entry in selected:
                key = (kind, tuple(entry["region"]))
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache_hits += 1
                    records.extend(cached)
                    continue
                self._cache_misses += 1
                stream.seek(entry["offset"])
                payload = stream.read(entry["byte_length"])
                if len(payload) != entry["byte_length"]:
                    raise ValueError("candidate data is truncated")
                bucket: list[CandidateRecord] = []
                for offset in range(0, len(payload), _RECORD.size):
                    x, y, z, kind_id, quality, rx, ry, rz = _RECORD.unpack_from(
                        payload, offset
                    )
                    bucket.append(
                        CandidateRecord(
                            position=(x, y, z),
                            kind=_ID_KIND[kind_id],
                            quality=quality,
                            region=(rx, ry, rz),
                        )
                    )
                self._cache[key] = tuple(bucket)
                records.extend(bucket)
        return records


def write_candidate_index(
    root: Path, world_identity: str, records: Sequence[CandidateRecord]
) -> None:
    """Write records grouped by region and kind for range-addressable reads."""

    ordered = sorted(
        records, key=lambda item: (item.region, _KIND_ID[item.kind], item.position)
    )
    entries: list[dict[str, object]] = []
    with root.joinpath(CANDIDATE_DATA_FILE).open("wb") as stream:
        start = 0
        group: list[CandidateRecord] = []
        group_key: tuple[tuple[int, int, int], CandidateKind] | None = None
        for record in (*ordered, None):
            key = None if record is None else (record.region, record.kind)
            if group and key != group_key:
                entries.append(
                    {
                        "region": list(group_key[0]),
                        "kind": group_key[1],
                        "offset": start,
                        "byte_length": len(group) * _RECORD.size,
                        "count": len(group),
                    }
                )
                start += len(group) * _RECORD.size
                group = []
            if record is not None:
                stream.write(
                    _RECORD.pack(
                        *record.position,
                        _KIND_ID[record.kind],
                        record.quality,
                        *record.region,
                    )
                )
                group.append(record)
                group_key = key
    data_sha256 = _sha256_file(root.joinpath(CANDIDATE_DATA_FILE))
    root.joinpath(CANDIDATE_INDEX_FILE).write_text(
        json.dumps(
            {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "world_identity_sha256": world_identity,
                "record_size": _RECORD.size,
                "data_sha256": data_sha256,
                "entries": entries,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(seed: int, stream: int, item: CandidateRecord) -> int:
    """FNV-1a rank mirrored by the native ABI without RNG state."""

    value = 0xCBF29CE484222325
    payload = seed.to_bytes(8, "little", signed=False) + stream.to_bytes(
        8, "little", signed=False
    )
    payload += b"".join(axis.to_bytes(4, "little") for axis in item.position)
    payload += bytes([_KIND_ID[item.kind]])
    for byte in payload:
        value ^= byte
        value = value * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
    return value
