"""Fresh data identities and calibration primitives for voxel pilot v2r1."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from theseo_anysearch.garden.pilots.contracts import FreshDrawIdentity, PoolIdentity
from theseo_anysearch.garden.pilots.v2 import V2_POOL_OBSERVATIONS, V2_POOL_SIZES
from theseo_anysearch.garden.splits import GeometryDescriptor, query_sha256


V2R1_DATASET_ID = "voxel-encoder-pilot-v2r1-dataset-1"
_FAMILIES = ("open", "thin_obstacle", "topology", "imported")
_BANDS = ("low", "medium", "high")


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _rank(seed: int, scope: str, geometry_id: str) -> str:
    return _canonical_sha({"seed": seed, "scope": scope, "geometry_id": geometry_id})


def v2r1_geometry_records() -> list[GeometryDescriptor]:
    """Return 248 identities disjoint from every opened v1/v2 geometry."""

    records: list[GeometryDescriptor] = []
    for index in range(216):
        family = _FAMILIES[index % len(_FAMILIES)]
        records.append(
            GeometryDescriptor(
                geometry_id=f"pilot-v2r1-{index:03d}",
                family=family,
                occupancy_band=_BANDS[(index // len(_FAMILIES)) % len(_BANDS)],
                source=(
                    "synthetic_mesh_import_fixture_v2r1"
                    if family == "imported"
                    else "procedural_voxel_fixture_v2r1"
                ),
            )
        )
    offset = 0
    for group, count in (("ordinary", 16), ("heldout_topology", 8), ("heldout_imported", 8)):
        for local_index in range(count):
            family = (
                "topology"
                if group == "heldout_topology"
                else "imported"
                if group == "heldout_imported"
                else _FAMILIES[local_index % len(_FAMILIES)]
            )
            records.append(
                GeometryDescriptor(
                    geometry_id=f"pilot-v2r1-confirm-{offset + local_index:02d}",
                    family=family,
                    occupancy_band=_BANDS[local_index % len(_BANDS)],
                    source=(
                        "heldout_synthetic_mesh_import_fixture_v2r1"
                        if family == "imported"
                        else "heldout_procedural_voxel_fixture_v2r1"
                    ),
                    confirmation_group=group,
                )
            )
        offset += count
    return records


def assign_v2r1_pools(
    records: Sequence[GeometryDescriptor], *, seed: int
) -> dict[str, tuple[str, ...]]:
    """Assign all fresh geometries with complete strata in each active pool."""

    if len(records) != 248 or len({record.geometry_id for record in records}) != 248:
        raise ValueError("v2r1 requires exactly 248 globally unique geometries")
    if any(not record.geometry_id.startswith("pilot-v2r1-") for record in records):
        raise ValueError("v2r1 cannot reuse a v1 or v2 geometry identity")
    regular = [record for record in records if record.confirmation_group is None]
    groups: dict[tuple[str, str], list[GeometryDescriptor]] = {}
    for record in regular:
        groups.setdefault((record.family, record.occupancy_band), []).append(record)
    expected = {(family, band) for family in _FAMILIES for band in _BANDS}
    if set(groups) != expected:
        raise ValueError("v2r1 regular geometries must cover all twelve strata")
    for key, values in groups.items():
        values.sort(key=lambda item: _rank(seed, f"v2r1:{key}", item.geometry_id))

    pools: dict[str, tuple[str, ...]] = {}
    for pool in (
        "pilot_train",
        "pilot_dev_early",
        "pilot_dev_arch",
        "pilot_dev_interaction",
        "pilot_calibration",
        "pilot_diagnostic",
    ):
        selected: list[GeometryDescriptor] = []
        while len(selected) < V2_POOL_SIZES[pool]:
            for key in sorted(groups):
                if groups[key] and len(selected) < V2_POOL_SIZES[pool]:
                    selected.append(groups[key].pop(0))
        pools[pool] = tuple(record.geometry_id for record in selected)

    confirmation: list[GeometryDescriptor] = []
    for group, count in (("ordinary", 16), ("heldout_topology", 8), ("heldout_imported", 8)):
        eligible = [record for record in records if record.confirmation_group == group]
        eligible.sort(key=lambda item: _rank(seed, f"v2r1:confirm:{group}", item.geometry_id))
        if len(eligible) != count:
            raise ValueError(f"v2r1 confirmation group {group} requires {count} geometries")
        confirmation.extend(eligible)
    pools["pilot_confirm"] = tuple(record.geometry_id for record in confirmation)
    assigned = [geometry_id for values in pools.values() for geometry_id in values]
    if len(assigned) != 248 or len(set(assigned)) != 248:
        raise ValueError("v2r1 pool assignment must use every geometry exactly once")
    return pools


def v2r1_query_plan(pool: str, geometry_ids: tuple[str, ...]) -> list[dict[str, object]]:
    """Freeze revised query families without opening their geometry contents."""

    if pool == "pilot_train":
        counts = (100_000, 50_000)
    elif pool == "pilot_confirm":
        counts = (40_000, 20_000)
    else:
        counts = (20_000, 10_000)
    return [
        {
            "pool": pool,
            "probe": probe,
            "count": count,
            "seed": 3290 + index,
            "geometry_ids_sha256": _canonical_sha(list(geometry_ids)),
            "assignment": "sha256_rank_round_robin_within_geometry_strata_v2r1",
            "revision": revision,
        }
        for index, (probe, count, revision) in enumerate(
            (
                ("coordinate", counts[0], "heldout_occupancy_v1"),
                ("pair", counts[1], "stratified_margin_boundary_v1"),
            )
        )
    ]


def build_v2r1_pool_identities(
    *, seed: int
) -> tuple[list[GeometryDescriptor], dict[str, PoolIdentity], dict[str, FreshDrawIdentity]]:
    records = v2r1_geometry_records()
    assigned = assign_v2r1_pools(records, seed=seed)
    pools = {
        pool: PoolIdentity(
            geometry_ids=geometry_ids,
            observations=V2_POOL_OBSERVATIONS[pool],
            assignment_sha256=_canonical_sha(
                {
                    "dataset": V2R1_DATASET_ID,
                    "seed": seed,
                    "pool": pool,
                    "geometry_ids": geometry_ids,
                }
            ),
            query_sha256=query_sha256(v2r1_query_plan(pool, geometry_ids)),
        )
        for pool, geometry_ids in assigned.items()
    }
    fresh_draws = {
        pilot: FreshDrawIdentity(
            seed=draw_seed,
            pool=pool,
            assignment_sha256=pools[pool].assignment_sha256,
            query_sha256=pools[pool].query_sha256,
        )
        for pilot, draw_seed, pool in (
            ("P4", 204, "pilot_dev_arch"),
            ("P6", 206, "pilot_dev_interaction"),
            ("P7", 207, "pilot_confirm"),
        )
    }
    return records, pools, fresh_draws


__all__ = [
    "V2R1_DATASET_ID",
    "assign_v2r1_pools",
    "build_v2r1_pool_identities",
    "v2r1_geometry_records",
    "v2r1_query_plan",
]
