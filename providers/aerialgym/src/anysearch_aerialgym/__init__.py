"""Optional source-pinned Aerial Gym derived-static world provider."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

from theseo_anysearch.environments.aerial_gym_export import export_scene
from theseo_anysearch.environments.routing_manifests import RightsRecord
from theseo_anysearch.world_providers.api import GenerationSummary, ProviderInfo, ProviderParameter
from theseo_anysearch.world_providers.source_cache import cached_sources

REVISION = "f0d0f05283f7897bab5a1bcc7b19b91cebbab218"
BASE_URL = f"https://raw.githubusercontent.com/ntnu-arl/aerial_gym_simulator/{REVISION}"
SOURCE_HASHES = {
    "LICENSE": "fa81cdc215b74bc8c905c9984c24f2d5810cd777e50dd8d0e6a0b64d9bd85440",
    "aerial_gym/config/env_config/env_with_lidar_nav_obstacles.py": "1123901601d73e6ac60ebd42bb273e2436c5c9e19dcdad9ad29a55ad1089c586",
    "aerial_gym/config/asset_config/lidar_nav_env_config.py": "b0dc710fee383b51ee0c6d98e4c9c80b213c530d96a5d1649ae9413faa0a3f4c",
    "resources/models/environment_assets/objects/1_x_1_wall.urdf": "3df34782da8e9279deab4e82fe72abe91498acb7d506baf064c017abe0c39080",
    "resources/models/environment_assets/walls/left_wall.urdf": "0ed96648d84f3d5dd8e4c7e657dcde88c3e6767b8ef4a87e57e333ade5233a7b",
    "resources/models/environment_assets/walls/right_wall.urdf": "7fad043f821981b9f056b07d772cea1826b4860d86e3259446e6fa30f9a7357b",
    "resources/models/environment_assets/walls/front_wall.urdf": "c2827c2b21a04de6d8cc4936ec33c77d3e75dda09b4a1d56b39aeb91b645898f",
    "resources/models/environment_assets/walls/back_wall.urdf": "3081eec3d1e009895daabca7a4c693899e5a6e74c356c3200d35a779159f0e51",
    "resources/models/environment_assets/walls/top_wall.urdf": "6182855161b8eaa174f28f773eab922f21c6dec95f9d937a3ed70e1fdb7f9fa5",
    "resources/models/environment_assets/walls/bottom_wall.urdf": "9d96aab5c1e1a7d11db3fb3a0c54faa60dc31564e009bc14bcf4be24f7e6d73b",
}


class Provider:
    info = ProviderInfo(
        name="aerialgym", version="0.1.0",
        description="Derived static collision-box scenes; not native simulator episodes",
        native_meters_per_voxel=0.25,
        parameters=(
            ProviderParameter("meters-per-voxel", "number", default=0.25, minimum=0.10, maximum=0.50),
            ProviderParameter("layout", "text", default="detour", help="detour or altitude"),
            ProviderParameter("body-radius-m", "number", default=0.25, minimum=0, maximum=0.50),
            ProviderParameter("offline", "boolean", default=False, help="use verified cache only"),
        ),
        resolution_parameter="meters-per-voxel",
    )

    def generate(self, *, seed: int, output: Path, parameters: dict[str, object]) -> GenerationSummary:
        if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
            raise ValueError("seed must be a uint32 integer")
        if output.exists():
            raise FileExistsError(output)
        allowed = {p.name: p for p in self.info.parameters}
        if set(parameters) - set(allowed):
            raise ValueError("unknown aerialgym parameters")
        resolved = {name: p.validate(parameters.get(name, p.default)) for name, p in allowed.items()}
        voxel_m = float(resolved["meters-per-voxel"])
        if not math.isfinite(voxel_m) or any(not math.isclose(length / voxel_m, round(length / voxel_m), abs_tol=1e-8) for length in (10, 6)):
            raise ValueError("meters-per-voxel must be finite and divide 10 and 6 meter bounds")
        if resolved["layout"] not in {"detour", "altitude"}:
            raise ValueError("layout must be detour or altitude")
        radius = float(resolved["body-radius-m"])
        if not math.isfinite(radius):
            raise ValueError("body radius must be finite")
        cache = Path(os.getenv("ANYSEARCH_AERIALGYM_CACHE", str(Path.home() / ".cache/anysearch/aerialgym")))
        source = cached_sources(cache, base_url=BASE_URL, hashes=SOURCE_HASHES, offline=bool(resolved["offline"]))
        rights = RightsRecord(
            status="reviewed", license_expression="BSD-3-Clause",
            allowed_uses=("training", "evaluation", "redistribution"),
            evidence=f"Allowlisted rigid box URDFs at {REVISION}: no conflicting headers or mesh references; repository LICENSE hash {SOURCE_HASHES['LICENSE']}; retain notice. Not a review of other upstream assets.",
        )
        export_scene(source, output, revision=REVISION, seed=seed, layout=str(resolved["layout"]),
                     voxel_m=voxel_m, body_radius_m=radius, source_hashes=SOURCE_HASHES,
                     rights=rights, partition="train", voxel_cube_routes=True)
        (output / "task.json").rename(output / "task-00.json")
        (output / "reference.json").rename(output / "reference-00.json")
        shutil.copyfile(source / "LICENSE", output / "LICENSE")
        return GenerationSummary()
