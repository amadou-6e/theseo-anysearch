"""Optional cached Gazebo maze importer; never runs downloaded simulator code."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from theseo_anysearch.environments.gazebo_maze_export import (
    ARCHIVES, UPSTREAM_SHA, _extent, export_maze,
)
from theseo_anysearch.environments.routing_manifests import (
    ConversionRecord, RoutingReferenceRecord, RoutingSplitRecord, RoutingTaskRecord,
    RoutingWorldRecord, SplitMember, read_sidecar, write_sidecar,
)
from theseo_anysearch.world_providers.api import GenerationSummary, ProviderInfo, ProviderParameter
from theseo_anysearch.world_providers.bundle import load_bundle
from theseo_anysearch.world_providers.source_cache import cached_sources

SPEC_SHA = "d98320117af66e3189d2b1b83b8f49897e91cfd2"
SOURCE_HASHES = {
    **ARCHIVES,
    "LICENSE": "be970277b34c9fffa57b5a08d718bfc6a981308258be6b6f43badafc48444d32",
}
BASE_URL = f"https://raw.githubusercontent.com/engcang/gazebo_maps/{UPSTREAM_SHA}"


def _updated(record, **changes):
    return type(record).model_validate({**record.model_dump(), **changes})


class Provider:
    info = ProviderInfo(
        name="gazebo", version="0.1.0",
        description="Static roofed SDF maze; evaluation-only, not native Gazebo episodes",
        native_meters_per_voxel=0.5,
        parameters=(
            ProviderParameter("meters-per-voxel", "number", default=0.5, minimum=0.25, maximum=0.5,
                              help="Regenerate SDF at 0.25-0.5 m; must divide 92/9 m bounds"),
            ProviderParameter("layout", "text", default="roofed",
                              help="roofed (derived ceiling at 8 m). Default: roofed"),
            ProviderParameter("body-radius-m", "number", default=0.25, minimum=0.01, maximum=0.49),
            ProviderParameter("offline", "boolean", default=False, help="Use verified cached archives only"),
        ),
        resolution_parameter="meters-per-voxel",
    )

    def generate(self, *, seed: int, output: Path, parameters: dict[str, object]) -> GenerationSummary:
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a uint32 integer")
        if output.exists():
            raise FileExistsError(output)
        allowed = {p.name: p for p in self.info.parameters}
        if set(parameters) - set(allowed):
            raise ValueError("unknown gazebo parameters")
        resolved = {name: p.validate(parameters.get(name, p.default)) for name, p in allowed.items()}
        voxel_m = float(resolved["meters-per-voxel"])
        _extent(voxel_m)
        if resolved["layout"] != "roofed":
            raise ValueError("layout must be roofed")
        if not math.isfinite(float(resolved["body-radius-m"])):
            raise ValueError("body radius must be finite")
        cache = Path(os.getenv("ANYSEARCH_GAZEBO_CACHE", str(Path.home() / ".cache/anysearch/gazebo")))
        source = cached_sources(cache, base_url=BASE_URL, hashes=SOURCE_HASHES,
                                offline=bool(resolved["offline"]), max_file_bytes=4 * 1024 * 1024)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gazebo-", dir=output.parent) as temporary:
            root = Path(temporary) / "world"
            report = export_maze(source, root, voxel_m=voxel_m,
                                 body_radius_m=float(resolved["body-radius-m"]), compile_packs=False)
            names = [row["name"] for row in report["accepted_queries"]]
            if not names:
                raise ValueError("no accepted Gazebo tasks")
            names = names[seed % len(names):] + names[:seed % len(names)]
            old_world = read_sidecar(root / "world-roofed.json", RoutingWorldRecord)
            old_conversion = read_sidecar(root / "conversion-roofed.json", ConversionRecord)
            conversion = _updated(old_conversion, converter_version="2-provider",
                parent_world_identity_sha256=None,
                parameters={**old_conversion.parameters,
                    "original_open_top_world_sha256": report["open_top_world_identity_sha256"]})
            world = _updated(old_world, conversion_identity_sha256=conversion.identity_sha256,
                            parent_world_identity_sha256=None)
            write_sidecar(root / "conversion.json", conversion)
            write_sidecar(root / "world.json", world)
            (root / "roofed-occupancy.npy").rename(root / "occupancy.npy")
            tasks = [read_sidecar(root / f"task-{name}.json", RoutingTaskRecord) for name in names]
            references = [read_sidecar(root / f"reference-{name}.json", RoutingReferenceRecord) for name in names]
            for path in list(root.glob("task-*.json")) + list(root.glob("reference-*.json")):
                path.unlink()
            for index, (task, reference) in enumerate(zip(tasks, references)):
                task = _updated(task, world_identity_sha256=world.identity_sha256)
                reference = _updated(reference, task_identity_sha256=task.identity_sha256)
                write_sidecar(root / f"task-{index:02d}.json", task)
                write_sidecar(root / f"reference-{index:02d}.json", reference)
            split = RoutingSplitRecord(dataset_id=f"gazebo-provider-v1-{UPSTREAM_SHA[:12]}",
                members=(SplitMember(world_identity_sha256=world.identity_sha256,
                    root_geometry_id=world.root_geometry_id, topology_family=world.topology_family,
                    site_id=world.site_id, partition="test"),))
            (root / "split.json").rename(root / "split-original.json")
            write_sidecar(root / "split.json", split)
            shutil.copyfile(source / "LICENSE", root / "LICENSE")
            report["provider_governing_spec_sha"] = SPEC_SHA
            report["provider_world_identity_sha256"] = world.identity_sha256
            report["seed_semantics"] = "rotates fixed accepted task ordering; static source geometry"
            report["seed"] = seed
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            load_bundle(root, use="evaluation")
            root.rename(output)
        return GenerationSummary(rejected_task_strata=tuple(
            f"{row['query']}:{row['reason']}" for row in report["rejected_queries"]))
