"""Immutable compiled-world variants selected by an episode seed."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class WorldVariant:
    root: Path
    identity_sha256: str
    portal_centers: tuple[tuple[int, int], ...]
    routes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SeededWorldCatalog:
    path: Path
    identity_sha256: str
    extent: tuple[int, int, int]
    variants: tuple[WorldVariant, ...]

    def for_seed(self, seed: int) -> WorldVariant:
        return self.variants[int(seed) % len(self.variants)]

    def route_for_stage(
        self, stage: int, seed: int, *, variation_radius: int = 0,
        action_mode: str = "discrete_18",
    ):
        from theseo_anysearch.rllib.trainer.waypoint_routes import (
            WaypointRoute, sample_fixed_route_variant,
        )

        variant = self.for_seed(seed)
        route = WaypointRoute.model_validate(variant.routes[stage])
        if variation_radius:
            route = sample_fixed_route_variant(
                route, radius=variation_radius, seed=seed,
                extent=self.extent, action_mode=action_mode,
            )
        return route


@lru_cache(maxsize=16)
def _load_catalog(resolved_path: Path) -> SeededWorldCatalog:
    from theseo_anysearch.worlds.compiler import validate_compiled_world

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    expected = payload.pop("catalog_sha256", None)
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if payload.get("schema_version") != 1 or expected != actual:
        raise ValueError("seeded world catalog version or checksum mismatch")
    extent = tuple(int(axis) for axis in payload["extent"])
    variants = []
    for entry in payload["variants"]:
        root = (resolved_path.parent / entry["path"]).resolve()
        compiled = validate_compiled_world(root)
        if (compiled.manifest.identity_sha256 != entry["world_identity_sha256"]
                or compiled.manifest.extent.as_tuple() != extent):
            raise ValueError("seeded world catalog pack identity or extent mismatch")
        variants.append(WorldVariant(
            root=compiled.root,
            identity_sha256=compiled.manifest.identity_sha256,
            portal_centers=tuple(tuple(point) for point in entry["portal_centers"]),
            routes=tuple(entry["routes"]),
        ))
    if len(variants) < 2 or len({v.identity_sha256 for v in variants}) != len(variants):
        raise ValueError("seeded world catalog requires distinct compiled variants")
    return SeededWorldCatalog(resolved_path, actual, extent, tuple(variants))


def load_catalog(path: str | Path) -> SeededWorldCatalog:
    return _load_catalog(Path(path).resolve())


def catalog_payload(extent: tuple[int, int, int], variants: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {"schema_version": 1, "extent": list(extent), "variants": variants}
    return {**payload, "catalog_sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
