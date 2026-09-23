"""Optional CaveDroneSim provider; upstream source stays outside both wheels."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

from theseo_anysearch.environments.cave_drone_export import (
    EXPECTED_EXTENT,
    EXPECTED_VOXEL_SIZE_M,
    GENERATOR_FAMILY,
    UPSTREAM_COMMIT,
    export_seed,
)
from theseo_anysearch.world_providers.api import (
    GenerationSummary,
    ProviderInfo,
    ProviderParameter,
)


class Provider:
    info = ProviderInfo(
        name="cavedrone",
        version="0.1.0",
        description="Pinned CaveDroneSim native chamber-and-tunnel voxel worlds",
        native_meters_per_voxel=EXPECTED_VOXEL_SIZE_M,
        native_extent=EXPECTED_EXTENT,
        parameters=(
            ProviderParameter(
                "partition", "text", default="train",
                help="Split role: train, validation, or test (default: train)",
            ),
            ProviderParameter(
                "body-radius-m", "number", default=0.25, minimum=0, maximum=2,
                help="Swept-sphere body radius in meters (default: 0.25)",
            ),
        ),
    )

    def generate(self, *, seed: int, output: Path, parameters: dict[str, object]) -> GenerationSummary:
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("CaveDrone seed must be a uint32 integer")
        if set(parameters) - {"partition", "body-radius-m"}:
            raise ValueError("unsupported CaveDrone provider parameter")
        partition = parameters.get("partition", "train")
        if partition not in {"train", "validation", "test"}:
            raise ValueError("partition must be train, validation, or test")
        radius = parameters.get("body-radius-m", 0.25)
        if type(radius) not in (int, float) or not math.isfinite(radius) or not 0 <= radius <= 2:
            raise ValueError("body-radius-m must be a finite number from 0 to 2")
        source_path = os.environ.get("ANYSEARCH_CAVEDRONE_SOURCE")
        if not source_path:
            raise ValueError(
                "set ANYSEARCH_CAVEDRONE_SOURCE to an explicitly checked-out "
                f"CaveDroneSim source at {UPSTREAM_COMMIT}; no source is downloaded"
            )
        compiler = os.environ.get("ANYSEARCH_CAVEDRONE_CXX", "g++")
        if not compiler:
            raise ValueError("ANYSEARCH_CAVEDRONE_CXX cannot be empty")
        try:
            report = export_seed(
                Path(source_path), output, seed=seed, partition=partition,
                body_radius_m=float(radius), compiler=compiler,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"")
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise ValueError(f"CaveDrone source check or C++23 bridge failed: {detail.strip() or exc}") from exc
        if report["upstream_commit"] != UPSTREAM_COMMIT or report["topology_family"] != GENERATOR_FAMILY:
            raise ValueError("CaveDrone export source or generator family disagrees with provider metadata")
        return GenerationSummary(
            rejected_task_strata=tuple(report["rejected_task_strata"]),
            limitations=(
                "one generator family; a different seed is not a held-out topology family",
                "full occupancy truth, not simulated sensor evidence",
                "voxel-cube route feasibility, not native continuous flight validation",
            ),
        )
