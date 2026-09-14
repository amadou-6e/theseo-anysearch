"""Prepare, but do not train on, rights-cleared imported routing exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .external_routing import (
    GOVERNING_SPEC_SHA,
    load_imported_worlds,
    prepare_routing_rows,
    write_prepared_dataset,
)


class ImportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    export: str = Field(min_length=1)
    source_root: str = Field(min_length=1)
    partition: Literal["train", "calibration", "test"]


class PreparationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    governing_spec_sha: Literal[GOVERNING_SPEC_SHA]
    dataset_id: str = Field(min_length=1)
    imports: tuple[ImportSpec, ...] = Field(min_length=1)
    test_topology_families: tuple[str, ...] = ()


def run_plan(plan_path: Path, output: Path, *, asset_root: Path | None = None) -> dict:
    """Resolve paths from a declared asset root and write an immutable corpus."""

    plan_path = Path(plan_path)
    plan = PreparationPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    base = Path(asset_root) if asset_root is not None else plan_path.parent
    loaded = [
        load_imported_worlds(
            (base / entry.export).resolve(),
            source_root=(base / entry.source_root).resolve(),
        )
        for entry in plan.imports
    ]
    imports = [item for group in loaded for item in group]
    partitions: dict[str, Literal["train", "calibration", "test"]] = {}
    for entry, group in zip(plan.imports, loaded):
        for item in group:
            root = item.world.root_geometry_id
            if root in partitions and partitions[root] != entry.partition:
                raise ValueError("one root geometry cannot cross partitions")
            partitions[root] = entry.partition
    prepared = prepare_routing_rows(
        imports,
        dataset_id=plan.dataset_id,
        partitions=partitions,
        test_topology_families=plan.test_topology_families,
    )
    return write_prepared_dataset(output, prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, help="Base for ignored source and export paths")
    args = parser.parse_args()
    print(json.dumps(run_plan(args.plan, args.output, asset_root=args.asset_root), sort_keys=True))


if __name__ == "__main__":
    main()
