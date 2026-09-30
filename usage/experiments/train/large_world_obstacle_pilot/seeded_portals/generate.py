"""Compile a finite catalog of seed-selected, staggered gate worlds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from theseo_anysearch.rllib.trainer.waypoint_routes import route_distance
from theseo_anysearch.worlds.compiler import compile_world
from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.worlds.seeded_catalog import catalog_payload
from usage.experiments.train.large_world_obstacle_pilot.curriculum import (
    STAGE_LENGTHS, gate_routes,
)
from usage.experiments.train.large_world_obstacle_pilot.preflight import (
    ACTION_MODE, EXTENT, PORTAL_CENTER, PORTAL_WALLS, wall_sources,
)

DEFAULT_VARIANTS = 16
GENERATOR_SEED = 437_000


def staggered_centers(layout_seed: int) -> tuple[tuple[int, int], ...]:
    """Alternate opposite sides of the centerline, with seeded offsets."""
    rng = random.Random(layout_seed)
    first_sign = 1 if layout_seed % 2 else -1
    return tuple(
        (PORTAL_CENTER[0] + first_sign * (-1) ** index * rng.randint(96, 144),
         PORTAL_CENTER[1])
        for index in range(len(PORTAL_WALLS))
    )


def build_catalog(output_dir: Path, variant_count: int = DEFAULT_VARIANTS) -> dict:
    if variant_count < 2:
        raise ValueError("at least two world variants are needed")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    identities = set()
    for index in range(variant_count):
        layout_seed = GENERATOR_SEED + index
        centers = staggered_centers(layout_seed)
        sources = tuple(
            box
            for (x, side), center in zip(PORTAL_WALLS, centers)
            for box in wall_sources(x, side, center)
        )
        compiled = compile_world(
            sources, WorldExtent.from_value(EXTENT), output_dir,
            generate_candidates=False,
        )
        identity = compiled.manifest.identity_sha256
        if identity in identities:
            raise ValueError("seeded layouts produced duplicate world packs")
        identities.add(identity)
        routes = gate_routes(portal_centers=centers)
        if any(route_distance(route, ACTION_MODE) != expected
               for route, expected in zip(routes, STAGE_LENGTHS)):
            raise ValueError(f"layout {layout_seed} violates exact stage lengths")
        variants.append({
            "layout_seed": layout_seed,
            "path": compiled.root.relative_to(output_dir).as_posix(),
            "world_identity_sha256": identity,
            "portal_centers": [list(center) for center in centers],
            "routes": [route.model_dump(mode="json") for route in routes],
        })
    catalog = catalog_payload(EXTENT, variants)
    catalog_path = output_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return {"path": str(catalog_path), "identity_sha256": catalog["catalog_sha256"],
            "variants": variant_count, "world_identities": sorted(identities)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runtime/seeded-gate-pilot"))
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS)
    args = parser.parse_args()
    print(json.dumps(build_catalog(args.output_dir, args.variants), indent=2))


if __name__ == "__main__":
    main()
