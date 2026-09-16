"""Optional cached IFC-Bench West Riverside Hospital MEP voxel-world provider."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

from theseo_anysearch.environments.ifc_bench_export import (
    FILE_HASHES,
    SOURCE_REVISION,
    export_discipline,
)
from theseo_anysearch.environments.routing_manifests import (
    RoutingReferenceRecord,
    RoutingTaskRecord,
    read_sidecar,
    write_sidecar,
)
from theseo_anysearch.world_providers.api import GenerationSummary, ProviderInfo, ProviderParameter
from theseo_anysearch.world_providers.source_cache import cached_sources

BASE_URL = (
    "https://huggingface.co/datasets/sylvainHellin/ifc-bench/resolve/"
    f"{SOURCE_REVISION}/projects/west_riverside_hospital"
)
SOURCE_HASHES = dict(FILE_HASHES)

# Fixed, non-user-selectable canonical crop per discipline. A real
# west_riverside_hospital discipline spans roughly 85 x 33 x 65 m; at this
# adapter's pipe-appropriate voxel floor that is far past the voxel cap (see
# ifc_bench_export's own module docstring), so this provider narrows each
# discipline to one previously validated, connected MEP region rather than
# exposing crop selection to callers. meters-per-voxel below follows from
# each crop's own real geometry -- it is not a free resampling choice.
CANONICAL_SCOPE = {
    "plumbing": {
        "storey_names": ("Level 1",),
        "crop_bounds_m": ((53.0, 97.2, 165.9), (53.6, 99.2, 166.4)),
        "meters_per_voxel": 0.01,
    },
    "electrical": {
        "storey_names": None,
        "crop_bounds_m": ((26.84, 63.10, 169.1), (26.96, 63.60, 170.3)),
        # Not 0.01395: the adapter derives this from min(radius)/2 on the
        # real pinned file, and generate_world's declared-vs-actual check is
        # an exact float comparison, so this must be the precise bit value
        # that computation actually produces (confirmed by direct run against
        # the real, hash-verified elec_ifc4.ifc), not the nearest-looking
        # decimal literal.
        "meters_per_voxel": 0.013949999999999999,
    },
}


def _rename_retrying(source: Path, target: Path, *, attempts: int = 5) -> None:
    """Windows occasionally holds a transient handle (antivirus/indexer scan)
    on a just-written directory for a moment after its last file closes;
    retry the rename briefly rather than failing an otherwise-complete,
    correctly-verified generation on a race outside this process's control.
    """

    for attempt in range(attempts):
        try:
            source.rename(target)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


class Provider:
    info = ProviderInfo(
        name="ifcbench", version="0.1.0",
        description=(
            "Fixed-crop West Riverside Hospital plumbing/electrical MEP routing; "
            "evaluation-only, not a native IFC-Bench-labeled query"
        ),
        native_meters_per_voxel=0.01,
        parameters=(
            ProviderParameter(
                "discipline", "text", required=True,
                help="plumbing or electrical",
            ),
            ProviderParameter(
                "meters-per-voxel", "number", required=True,
                help=(
                    "Must equal this discipline's fixed, geometry-derived "
                    "resolution for the canonical crop -- plumbing: 0.01, "
                    "electrical: 0.013949999999999999 (exact float; see "
                    "README). Not a free resampling choice: generation "
                    "fails if it disagrees with what the adapter actually "
                    "derives from the source geometry."
                ),
            ),
            ProviderParameter("body-radius-m", "number", default=0.02, minimum=0, maximum=0.5),
            ProviderParameter("offline", "boolean", default=False, help="use verified cache only"),
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
            raise ValueError("unknown ifcbench parameters")
        resolved = {name: p.validate(parameters.get(name, p.default)) for name, p in allowed.items()}
        discipline = str(resolved["discipline"])
        if discipline not in CANONICAL_SCOPE:
            raise ValueError("discipline must be plumbing or electrical")
        scope = CANONICAL_SCOPE[discipline]
        declared_mpv = float(resolved["meters-per-voxel"])
        if declared_mpv != scope["meters_per_voxel"]:
            raise ValueError(
                f"meters-per-voxel for {discipline} must be {scope['meters_per_voxel']}, "
                "the fixed geometry-derived resolution for this discipline's canonical crop"
            )
        body_radius = float(resolved["body-radius-m"])
        if not math.isfinite(body_radius):
            raise ValueError("body radius must be finite")

        cache = Path(os.getenv("ANYSEARCH_IFCBENCH_CACHE", str(Path.home() / ".cache/anysearch/ifcbench")))
        source = cached_sources(
            cache, base_url=BASE_URL, hashes=SOURCE_HASHES,
            offline=bool(resolved["offline"]), max_file_bytes=32 * 1024 * 1024,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        # ignore_cleanup_errors: after root.rename(output) below, the now-empty
        # `temporary` directory occasionally still looks "in use" to Windows
        # for a moment (antivirus/indexer scan); the renamed data in `output`
        # is unaffected either way, so a failure to remove the leftover empty
        # directory must not fail generation.
        with tempfile.TemporaryDirectory(
            prefix="ifcbench-", dir=output.parent, ignore_cleanup_errors=True
        ) as temporary:
            root = Path(temporary) / "world"
            report = export_discipline(
                source, root, discipline=discipline, partition="test",
                body_radius_m=body_radius,
                storey_names=scope["storey_names"], crop_bounds_m=scope["crop_bounds_m"],
                # cached_sources() above already verified these exact pinned
                # hashes moments ago; the adapter's own re-check exists for
                # direct/standalone adapter callers that bypass the cache.
                verify_hashes=False,
            )
            task_paths = sorted(root.glob("task-*.json"))
            reference_paths = sorted(root.glob("reference-*.json"))
            count = len(task_paths)
            if count == 0:
                raise ValueError(f"no accepted {discipline} tasks for the fixed canonical crop")
            tasks = [read_sidecar(path, RoutingTaskRecord) for path in task_paths]
            references = [read_sidecar(path, RoutingReferenceRecord) for path in reference_paths]
            # Seed rotates the fixed accepted-task roster's order; it does
            # not alter which tasks are accepted or the static source
            # geometry, matching the Gazebo provider's seed semantics.
            offset = seed % count
            order = list(range(offset, count)) + list(range(offset))
            for path in [*task_paths, *reference_paths]:
                path.unlink()
            for new_index, old_index in enumerate(order):
                write_sidecar(root / f"task-{new_index:02d}.json", tasks[old_index])
                write_sidecar(root / f"reference-{new_index:02d}.json", references[old_index])
            # The adapter's report arrays are index-aligned with the
            # pre-rotation task/reference order; reorder them the same way
            # so the report agrees with the rewritten task-NN.json files
            # and the provider's own selected-task ordering, not the
            # adapter's original order.
            for key in ("task_ids", "reference_ids", "routes"):
                if key in report and len(report[key]) == count:
                    report[key] = [report[key][old_index] for old_index in order]
            report["provider_seed"] = seed
            report["seed_semantics"] = "rotates fixed accepted task ordering; static source geometry"
            (root / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            shutil.copyfile(source / "license.txt", root / "license.txt")
            _rename_retrying(root, output)
        return GenerationSummary(
            rejected_task_strata=tuple(report.get("rejected_task_strata", ())),
        )
