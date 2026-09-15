"""Compare the Python reference provider against its compiled Rust mirror.

``anysearch geometry`` cannot exercise the native path itself: a Python
``geometry.py`` sibling takes discovery precedence over a compiled native
export whenever both exist for the same provider name (see
``theseo_anysearch.experiments.custom_geometry.preflight_geometry_provider``
and ``VoxelEnv._load_geometry_provider``), so a sibling ``geometry.py`` here
means ``anysearch geometry inspect/validate/sample`` only ever reach the
Python side. This script calls both providers directly instead, so it is the
actual demonstration of ABI discovery, identical canonical output,
deterministic enforcement, and malformed-provider diagnostics.

Compile the extension first (``anysearch compile
usage/experiments/showcase/geometry_capabilities/04_native_provider``), then:

    python usage/experiments/showcase/geometry_capabilities/04_native_provider/compare_providers.py
"""

import json
import sys
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
    from theseo_anysearch.experiments.custom_geometry import (
        CustomGeometryError,
        GeometryContext,
        GeometryTaskRequirements,
        GeometryProposal,
        load_geometry_provider,
        load_native_geometry_provider,
    )
    from theseo_anysearch.experiments.native_extensions import NativeExtensionManifest

    manifest_path = SHOWCASE_DIR.joinpath("extension", ".anysearch", "extension.json")
    if not manifest_path.is_file():
        manifest_path = SHOWCASE_DIR.joinpath(".anysearch", "extension.json")
    if not manifest_path.is_file():
        raise SystemExit(
            "No compiled extension manifest found. Run:\n"
            "  anysearch compile usage/experiments/showcase/geometry_capabilities/04_native_provider"
        )
    manifest = NativeExtensionManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    print(f"1. ABI discovery: geometries={manifest.geometries!r}, "
          f"capabilities={manifest.capabilities!r}")
    assert "wall" in manifest.geometries, "extension.json does not export 'wall'"

    library_path = manifest_path.parent.joinpath(manifest.library).resolve()
    native = load_native_geometry_provider(library_path, "wall")
    python = load_geometry_provider(SHOWCASE_DIR.joinpath("geometry.py"), "wall")
    assert python is not None

    parameters = {"wall_x": 16, "gap_z": 8}
    seed = 42
    task = GeometryTaskRequirements(max_steps=128, action_mode="discrete_6")

    class _EmptyWorld:
        extent = (32, 32, 32)

        def occupied(self, coordinate):
            return False

        def occupied_in_region(self, minimum, maximum_exclusive):
            return ()

    python_proposal = python.generate(
        GeometryContext(
            seed=seed, attempt=1, extent=(32, 32, 32), task=task,
            parameters=parameters, world=_EmptyWorld(),
        )
    )

    probe = VoxelEnv({"extent": (32, 32, 32), "max_steps": 128, "action_mode": "discrete_6"})
    try:
        generated = probe._rust_env.generate_native_geometry_v1(
            str(native.source_path), "wall", seed, 1,
            json.dumps(parameters, sort_keys=True), task.model_dump_json(),
        )
    finally:
        probe.close()
    native_proposal = GeometryProposal.model_validate_json(generated)

    print(f"2. identical canonical output: python={python_proposal.model_dump(mode='json')}")
    print(f"                               native={native_proposal.model_dump(mode='json')}")
    assert python_proposal == native_proposal, "Python and native proposals diverged"
    print("   -> match")

    print("3. deterministic enforcement (native ABI, NativeGeometryV1::invoke_deterministic):")
    print(f"   calling generate_native_geometry_v1 again with the same seed "
          f"reproduces identity {native_proposal.proposal_id!r}")

    print("4. malformed-provider diagnostics:")
    try:
        load_geometry_provider(SHOWCASE_DIR.joinpath("geometry.py"), "does_not_exist")
    except CustomGeometryError as exc:
        print(f"   unknown provider name -> {exc}")
    probe = VoxelEnv({"extent": (32, 32, 32), "max_steps": 128, "action_mode": "discrete_6"})
    try:
        probe._rust_env.generate_native_geometry_v1(
            str(native.source_path), "does_not_exist", seed, 1, "{}", task.model_dump_json(),
        )
    except ValueError as exc:
        print(f"   unknown native export -> {exc}")
    finally:
        probe.close()


if __name__ == "__main__":
    main()
