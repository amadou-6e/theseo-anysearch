"""Deterministic, geometry-disjoint split construction for encoder pilots."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Sequence


GeometryFamily = Literal["open", "thin_obstacle", "topology", "imported"]
OccupancyBand = Literal["low", "medium", "high"]
ConfirmationGroup = Literal["ordinary", "heldout_topology", "heldout_imported"]

POOL_SIZES = {
    "pilot_train": 96,
    "pilot_dev_early": 24,
    "pilot_dev_arch": 24,
    "pilot_dev_interaction": 24,
    "pilot_confirm": 32,
}
POOL_OBSERVATIONS = {
    "pilot_train": 24_000,
    "pilot_dev_early": 6_000,
    "pilot_dev_arch": 6_000,
    "pilot_dev_interaction": 6_000,
    "pilot_confirm": 12_000,
}
CONFIRMATION_COUNTS = {
    "ordinary": 16,
    "heldout_topology": 8,
    "heldout_imported": 8,
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _stable_rank(seed: int, scope: str, geometry_id: str) -> str:
    return _canonical_hash({"seed": seed, "scope": scope, "geometry_id": geometry_id})


@dataclass(frozen=True)
class GeometryDescriptor:
    """Metadata assigned before any observations are sampled."""

    geometry_id: str
    family: GeometryFamily
    occupancy_band: OccupancyBand
    source: str
    parent_split: Literal["train"] = "train"
    confirmation_group: ConfirmationGroup | None = None

    def __post_init__(self) -> None:
        if not self.geometry_id or not self.source:
            raise ValueError("geometry_id and source cannot be empty")
        if self.family not in {"open", "thin_obstacle", "topology", "imported"}:
            raise ValueError(f"unsupported geometry family: {self.family}")
        if self.occupancy_band not in {"low", "medium", "high"}:
            raise ValueError(f"unsupported occupancy band: {self.occupancy_band}")
        if self.parent_split != "train":
            raise ValueError("pilot geometries must come from the parent training split")
        if self.confirmation_group not in {
            None,
            "ordinary",
            "heldout_topology",
            "heldout_imported",
        }:
            raise ValueError(f"unsupported confirmation group: {self.confirmation_group}")


@dataclass(frozen=True)
class PoolAssignment:
    """Frozen identities for all named pilot pools."""

    pools: dict[str, tuple[str, ...]]
    assignment_sha256: str
    records_sha256: str

    def pool_for(self, geometry_id: str) -> str:
        matches = [name for name, ids in self.pools.items() if geometry_id in ids]
        if len(matches) != 1:
            raise KeyError(f"geometry {geometry_id!r} is assigned to {len(matches)} pools")
        return matches[0]


@dataclass(frozen=True)
class FreshPoolDraw:
    """Identity of a pool that remains closed until a specified pilot."""

    pilot: Literal["P4", "P6", "P7"]
    pool: Literal["pilot_dev_arch", "pilot_dev_interaction", "pilot_confirm"]
    seed: int
    assignment_sha256: str
    query_sha256: str


def _balanced_take(
    candidates: Sequence[GeometryDescriptor], count: int, *, seed: int, scope: str
) -> list[GeometryDescriptor]:
    """Take a deterministic, near-balanced sample across family/density strata."""

    groups: dict[tuple[str, str], list[GeometryDescriptor]] = {}
    for record in candidates:
        groups.setdefault((record.family, record.occupancy_band), []).append(record)
    for records in groups.values():
        records.sort(key=lambda item: _stable_rank(seed, scope, item.geometry_id))

    families = {record.family for record in candidates}
    bands = {record.occupancy_band for record in candidates}
    if families != {"open", "thin_obstacle", "topology", "imported"}:
        raise ValueError(f"{scope} candidates do not cover every geometry family")
    if bands != {"low", "medium", "high"}:
        raise ValueError(f"{scope} candidates do not cover every occupancy band")

    selected: list[GeometryDescriptor] = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"not enough geometries to fill {scope} ({count} required)")
    return selected


def _validate_pool_coverage(
    pool: str, geometry_ids: Sequence[str], by_id: dict[str, GeometryDescriptor]
) -> None:
    records = [by_id[geometry_id] for geometry_id in geometry_ids]
    if {record.family for record in records} != {"open", "thin_obstacle", "topology", "imported"}:
        raise ValueError(f"{pool} does not cover every geometry family")
    if {record.occupancy_band for record in records} != {"low", "medium", "high"}:
        raise ValueError(f"{pool} does not cover every occupancy band")


def assign_pilot_pools(
    records: Sequence[GeometryDescriptor], *, seed: int
) -> PoolAssignment:
    """Assign complete geometries to the five immutable pilot pools."""

    if len(records) < sum(POOL_SIZES.values()):
        raise ValueError("at least 200 parent-training geometries are required")
    by_id = {record.geometry_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("geometry IDs must be globally unique before assignment")
    if any(record.parent_split != "train" for record in records):
        raise ValueError("pilot geometries must come only from the parent training split")

    confirmation: list[GeometryDescriptor] = []
    for group, count in CONFIRMATION_COUNTS.items():
        eligible = [record for record in records if record.confirmation_group == group]
        eligible.sort(key=lambda item: _stable_rank(seed, f"pilot_confirm:{group}", item.geometry_id))
        if len(eligible) < count:
            raise ValueError(f"pilot_confirm requires {count} geometries from {group}")
        confirmation.extend(eligible[:count])

    remaining = [record for record in records if record.confirmation_group is None]
    pools: dict[str, tuple[str, ...]] = {}
    for pool in ("pilot_train", "pilot_dev_early", "pilot_dev_arch", "pilot_dev_interaction"):
        selected = _balanced_take(remaining, POOL_SIZES[pool], seed=seed, scope=pool)
        pools[pool] = tuple(record.geometry_id for record in selected)
        selected_ids = set(pools[pool])
        remaining = [record for record in remaining if record.geometry_id not in selected_ids]
    pools["pilot_confirm"] = tuple(record.geometry_id for record in confirmation)

    all_assigned = [geometry_id for values in pools.values() for geometry_id in values]
    if len(all_assigned) != len(set(all_assigned)):
        raise ValueError("pilot pools overlap")
    for pool, expected_count in POOL_SIZES.items():
        if len(pools[pool]) != expected_count:
            raise ValueError(f"{pool} must contain {expected_count} geometries")
        _validate_pool_coverage(pool, pools[pool], by_id)

    confirmation_records = [by_id[geometry_id] for geometry_id in pools["pilot_confirm"]]
    actual_groups = {
        group: sum(record.confirmation_group == group for record in confirmation_records)
        for group in CONFIRMATION_COUNTS
    }
    if actual_groups != CONFIRMATION_COUNTS:
        raise ValueError("pilot_confirm must have the frozen 16/8/8 composition")

    record_payload = [
        {
            "geometry_id": record.geometry_id,
            "family": record.family,
            "occupancy_band": record.occupancy_band,
            "source": record.source,
            "parent_split": record.parent_split,
            "confirmation_group": record.confirmation_group,
        }
        for record in sorted(records, key=lambda item: item.geometry_id)
    ]
    assignment_payload = {
        "seed": seed,
        "pools": {pool: list(ids) for pool, ids in sorted(pools.items())},
        "observations": POOL_OBSERVATIONS,
    }
    return PoolAssignment(
        pools=pools,
        assignment_sha256=_canonical_hash(assignment_payload),
        records_sha256=_canonical_hash(record_payload),
    )


def query_sha256(queries: Sequence[dict[str, object]]) -> str:
    """Hash sorted immutable query identities and stratum weights."""

    normalized = sorted(queries, key=lambda item: _canonical_hash(item))
    return _canonical_hash(normalized)


def make_fresh_draws(
    assignment: PoolAssignment,
    queries_by_pool: dict[str, Sequence[dict[str, object]]],
    *,
    seeds: dict[str, int],
) -> dict[str, FreshPoolDraw]:
    """Create reproducible identities for the pools first opened at P4/P6/P7."""

    mapping = {
        "P4": "pilot_dev_arch",
        "P6": "pilot_dev_interaction",
        "P7": "pilot_confirm",
    }
    if set(seeds) != set(mapping):
        raise ValueError("fresh-draw seeds are required exactly for P4, P6, and P7")
    draws: dict[str, FreshPoolDraw] = {}
    for pilot, pool in mapping.items():
        if pool not in queries_by_pool:
            raise ValueError(f"missing frozen queries for {pool}")
        pool_assignment_hash = _canonical_hash(
            {"pool": pool, "geometry_ids": list(assignment.pools[pool]), "seed": seeds[pilot]}
        )
        draws[pilot] = FreshPoolDraw(
            pilot=pilot,
            pool=pool,
            seed=seeds[pilot],
            assignment_sha256=pool_assignment_hash,
            query_sha256=query_sha256(queries_by_pool[pool]),
        )
    return draws
