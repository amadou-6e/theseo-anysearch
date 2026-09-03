# Native geometry provider

`extension/src/lib.rs` exports a `wall` geometry provider via the
`#[anysearch_geometry_v1]` macro, mirroring the Python reference in the
sibling `geometry.py` exactly: same wall-with-a-gap logic, driven only by
`context.parameters`/`context.seed` (no randomness), so the two are
byte-comparable.

```powershell
anysearch compile usage/experiments/showcase/geometry_capabilities/04_native_provider
python usage/experiments/showcase/geometry_capabilities/04_native_provider/compare_providers.py
```

**`anysearch geometry inspect/validate/sample` do not exercise the native
path here.** A Python `geometry.py` sibling takes discovery precedence over a
compiled native export whenever both exist for the same provider name (see
`preflight_geometry_provider` / `VoxelEnv._load_geometry_provider`) --
exactly the same precedence a live training run would use. So `compare_providers.py`
is the actual demonstration; it calls both providers directly and confirms,
independently:

1. **ABI discovery** -- reads the compiled `extension.json` and asserts
   `"wall"` is in `manifest.geometries`.
2. **Identical canonical output** -- generates a proposal from each provider
   for the same seed/parameters and asserts the parsed `GeometryProposal`s
   are equal (verified: both return `sources: [[16,1,1,16,30,7], [16,1,9,16,30,30]]`
   for `wall_x=16, gap_z=8, seed=42`).
3. **Deterministic enforcement** -- the native ABI itself enforces this
   (`NativeGeometryV1::invoke_deterministic` in
   `theseo_anysearch/core/src/voxel/geometries/loader.rs` calls the provider
   twice per invocation and rejects a mismatch), and
   `preflight_geometry_provider` triggers that check once at preflight
   against a throwaway probe environment, before Ray starts, the same point
   a nondeterministic Python provider already fails at.
4. **Malformed-provider diagnostics** -- requesting an unexported name from
   each side surfaces the actual, distinct error each layer raises (a
   missing-callable `CustomGeometryError` from Python, a `GetProcAddress`
   failure surfaced as a `ValueError` from the native loader).

`anysearch geometry validate` on `experiment.yaml` still works and reports
`geometry: valid` / `task: feasible` -- it is just validating the Python
`wall`, not the Rust one.

## A note on Windows path length

Compiling this extension nests a dependency's own build-script path
(`extension/target/release/build/<crate>-<hash>/build_script_build-<hash>.exe`)
under `usage/experiments/showcase/geometry_capabilities/04_native_provider/`.
On a typical single clone this stays comfortably under Windows' 260-character
`MAX_PATH`; inside a deeply nested worktree layout (for example
`...\.worktrees\<branch>\...`) it can exceed it, and `cargo build` fails with
a linker error claiming it cannot open its own build-script `.exe`. If you
hit that, either enable Windows long-path support (`git config --system
core.longpaths true`, plus the matching `LongPathsEnabled` registry/group
policy setting) or clone/checkout this worktree closer to a drive root.
