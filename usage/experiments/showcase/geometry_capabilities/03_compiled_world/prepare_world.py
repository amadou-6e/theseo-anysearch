"""Compile the two moderate non-cubic worlds this showcase step inspects.

``anysearch geometry`` and ``anysearch compile`` never generate compiled-world
packs themselves -- they only inspect ones that already exist on disk. Run
this script once before using ``experiment_valid.yaml`` /
``experiment_blocked.yaml``:

    python usage/experiments/showcase/geometry_capabilities/03_compiled_world/prepare_world.py

It writes both packs under ``runtime/geometry_capabilities/03_compiled_world/``
(gitignored, not committed) via the same content-addressed ``compile_world``
API a training run would use, so nothing here is showcase-only plumbing. Each
pack lands in its own identity-hashed subdirectory (that is how
``compile_world``'s content-addressed cache works), so this also rewrites the
``compiled_world_path`` placeholder in ``experiment_valid.yaml`` /
``experiment_blocked.yaml`` to point at the exact path just compiled.

Both worlds share one 128x96x64 extent and one 2-voxel-thick dividing wall at
x=64-65. The valid pack leaves a gap in the wall at y=41-56; the blocked pack
fills it in completely, so the same start/goal pair that is reachable in one
is provably unreachable in the other -- exactly what
``anysearch geometry validate`` is meant to catch.

Bonus, clearly separate from the two packs above: sparse compiled-world
transformations (added in #325 / feat/306) are not yet exposed as experiment
YAML or an `anysearch geometry` command -- there is no config field or CLI
flag that installs a `SparseBoxTransform` today. The block at the bottom of
this script calls that programmatic API directly against the *valid* pack to
show what it does: layering a deterministic, content-addressed obstacle over
an existing compiled world without touching its immutable base. That call is
demonstration glue, not something `anysearch geometry` runs for you yet.
"""

import os
import re
from pathlib import Path

from theseo_anysearch.worlds.compiler import BoxSource, WorldCompilerConfig, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.worlds.transformations import (
    SparseTransformedRead,
    generate_box_transform,
)

EXTENT = WorldExtent(x=128, y=96, z=64)
WALL_X = (63, 64)  # zero-based, inclusive -- storage convention
GAP_Y = (40, 55)  # zero-based, inclusive
SHOWCASE_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = SHOWCASE_DIR.parents[4].joinpath(
    "runtime", "geometry_capabilities", "03_compiled_world"
)
PLACEHOLDER = "PREPARE_WORLD_PY_FILLS_THIS_IN"


def _wall_sources(*, with_gap: bool) -> list[BoxSource]:
    if not with_gap:
        return [BoxSource((WALL_X[0], 0, 0), (WALL_X[1], 95, 63))]
    sources = []
    if GAP_Y[0] > 0:
        sources.append(BoxSource((WALL_X[0], 0, 0), (WALL_X[1], GAP_Y[0] - 1, 63)))
    if GAP_Y[1] < 95:
        sources.append(BoxSource((WALL_X[0], GAP_Y[1] + 1, 0), (WALL_X[1], 95, 63)))
    return sources


def _point_experiment_at_pack(yaml_name: str, pack_root: Path) -> None:
    """Rewrite the compiled_world_path placeholder with the real, just-compiled path."""
    yaml_path = SHOWCASE_DIR.joinpath(yaml_name)
    relative = os.path.relpath(pack_root, start=SHOWCASE_DIR)
    content = yaml_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        rf"compiled_world_path: (?:{re.escape(PLACEHOLDER)}|\.\..*)",
        f"compiled_world_path: {Path(relative).as_posix()}",
        content,
    )
    if count != 1:
        raise RuntimeError(f"expected exactly one compiled_world_path line in {yaml_path}")
    yaml_path.write_text(updated, encoding="utf-8")
    print(f"{yaml_name}: compiled_world_path -> {relative}")


def main() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    config = WorldCompilerConfig(chunk_shape=(32, 32, 32))

    valid = compile_world(
        _wall_sources(with_gap=True), EXTENT, RUNTIME_ROOT.joinpath("valid"), config
    )
    print(f"valid pack:   {valid.root}")
    _point_experiment_at_pack("experiment_valid.yaml", valid.root)

    blocked = compile_world(
        _wall_sources(with_gap=False), EXTENT, RUNTIME_ROOT.joinpath("blocked"), config
    )
    print(f"blocked pack: {blocked.root}")
    _point_experiment_at_pack("experiment_blocked.yaml", blocked.root)

    # --- Bonus: the sparse-transformation API, called directly -------------
    # Not wired to any config field or CLI command yet; see the module
    # docstring above. This overlays a second, independently seeded wall on
    # top of the *valid* pack's already-compiled occupancy, entirely in
    # memory, without touching `valid.root` on disk. It only exercises point
    # queries: `GeometryArtifactRead` (from `GeometryArtifact.bounded_reader`)
    # doesn't expose a bounded region reader yet, so the region callable
    # below is never actually called and is left unimplemented on purpose.
    from theseo_anysearch.worlds.artifacts import load_geometry_artifact

    def _region_not_available(minimum, maximum_exclusive):
        raise NotImplementedError(
            "GeometryArtifactRead has no bounded region reader yet; "
            "this demo only exercises point queries"
        )

    valid_artifact = load_geometry_artifact(valid.root)
    transform = generate_box_transform(
        valid_artifact.manifest.identity_sha256, EXTENT, seed=7, count=1,
        minimum_size=(2, 20, 64), maximum_size=(2, 20, 64),
    )
    base_reader = valid_artifact.bounded_reader()
    overlay = SparseTransformedRead(base_reader.occupied, _region_not_available, transform)
    box = transform.boxes[0]
    sample_point = (box.minimum[0] + 1, box.minimum[1] + 1, 32)
    print(
        "sparse overlay demo: point "
        f"{sample_point} occupied under the overlay = {overlay.occupied(sample_point)} "
        f"(base pack alone: {base_reader.occupied(sample_point)})"
    )


if __name__ == "__main__":
    main()
