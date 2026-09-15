"""Small deterministic world used to exercise provider infrastructure offline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from theseo_anysearch.environments.routing_manifests import (
    ArtifactRef,
    ConversionRecord,
    GridFrame,
    RightsRecord,
    RoutingReferenceRecord,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceFile,
    SourceRecord,
    SplitMember,
    write_sidecar,
)
from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.world_providers.api import GenerationSummary, ProviderInfo, ProviderParameter


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixtureBoxesProvider:
    """Public-domain-style built-in test geometry, not a research corpus."""

    info = ProviderInfo(
        name="fixture-boxes",
        version="1",
        description="Deterministic 12-cube box-and-detour fixture (infrastructure test)",
        native_meters_per_voxel=1.0,
        native_extent=(12, 12, 12),
        parameters=(ProviderParameter("wall-height", "integer", default=3, minimum=1, maximum=5),),
    )

    def generate(self, *, seed: int, output: Path, parameters: dict[str, object]) -> GenerationSummary:
        if output.exists():
            raise FileExistsError(f"world output already exists: {output}")
        height = int(parameters.get("wall-height", 3))
        output.mkdir(parents=True)
        generator_path = output / "generator.json"
        generator_path.write_text(
            json.dumps({"seed": seed, "wall_height": height}, sort_keys=True), encoding="utf-8"
        )
        source = SourceRecord(
            source_id="fixture-boxes",
            source_url="builtin:fixture-boxes",
            revision="1",
            files=(SourceFile(relative_path="generator.json", sha256=_sha(generator_path), role="geometry"),),
            rights=RightsRecord(
                status="reviewed",
                license_expression="MIT",
                allowed_uses=("evaluation", "training"),
                evidence="Repository-owned synthetic fixture; not third-party source",
            ),
        )
        occupancy = np.zeros((12, 12, 12), dtype=np.uint8)
        occupancy[5, 2 : 2 + height, 1:9] = 1
        occupancy_path = output / "occupancy.npy"
        np.save(occupancy_path, occupancy, allow_pickle=False)
        occupancy_sha = _sha(occupancy_path)
        conversion = ConversionRecord(
            source_content_identity_sha256=source.content_identity_sha256,
            converter_name="fixture-boxes",
            converter_version="1",
            parameters={"seed": seed, "wall_height": height},
            output_occupancy_sha256=occupancy_sha,
        )
        world = RoutingWorldRecord(
            source_content_identity_sha256=source.content_identity_sha256,
            conversion_identity_sha256=conversion.identity_sha256,
            occupancy_sha256=occupancy_sha,
            extent=WorldExtent(x=12, y=12, z=12),
            frame=GridFrame(source_origin_m=(0.0, 0.0, 0.0), meters_per_voxel=1.0),
            root_geometry_id=f"fixture-boxes-v1-seed-{seed}",
            topology_family="fixture-box-detour",
        )
        route = (
            [(2, y, 2) for y in range(2, 6)]
            + [(x, 5, 2) for x in range(3, 9)]
            + [(8, y, 2) for y in range(4, 1, -1)]
        )
        task = RoutingTaskRecord(
            world_identity_sha256=world.identity_sha256,
            provenance="derived",
            family="point_path",
            start_storage=route[0],
            goal_storage=route[-1],
            movement_model="6-axis voxel-center route; occupancy-cube replay",
            body_radius_m=0.0,
            derivation_reason="Detour fixture exercises independently checked route replay",
        )
        route_path = output / "route-00.json"
        route_path.write_text(json.dumps(route, separators=(",", ":")), encoding="utf-8")
        reference = RoutingReferenceRecord(
            task_identity_sha256=task.identity_sha256,
            claim="independently_validated",
            route_artifact=ArtifactRef(relative_path=route_path.name, sha256=_sha(route_path)),
            cost=len(route) - 1,
            verification_evidence="Six-axis centerlines replayed against full occupied voxel cubes",
        )
        split = RoutingSplitRecord(
            dataset_id=f"fixture-boxes-v1-seed-{seed}",
            members=(SplitMember(
                world_identity_sha256=world.identity_sha256,
                root_geometry_id=world.root_geometry_id,
                topology_family=world.topology_family,
                partition="train",
            ),),
        )
        for name, record in (
            ("source.json", source), ("conversion.json", conversion),
            ("world.json", world), ("task-00.json", task),
            ("reference-00.json", reference), ("split.json", split),
        ):
            write_sidecar(output / name, record)
        return GenerationSummary()
