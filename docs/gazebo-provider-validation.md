# Cached Gazebo provider validation (#467)

Governing spec: `amadou-6e/specs@d98320117af66e3189d2b1b83b8f49897e91cfd2`
(`gazebo-world-provider.md`, specs PR #98). Branch `feat/467` stacks on
`feat/466@91de03e` / PR #470. Owner explicitly authorized unmerged stacking;
this does not authorize merging. Disposition: retain shared infrastructure.

## Local installation

Build core and optional provider separately; nothing is uploaded:

```powershell
python -m pip wheel --no-deps --wheel-dir runtime/wheels .
python -m pip wheel --no-deps --wheel-dir runtime/wheels ./providers/gazebo
python -m pip install --find-links runtime/wheels "theseo-anysearch[gazebo]==0.1.0"
anysearch worlds gazebo --help
anysearch worlds gazebo --seed 42 --meters-per-voxel 0.5 --output worlds/maze-42
anysearch worlds gazebo --seed 42 --meters-per-voxel 0.25 --offline true --output worlds/maze-42-fine
```

If replacing a local 0.1.0 core wheel, reinstall that wheel explicitly with
`--force-reinstall --no-deps` first; ordinary pip installation may otherwise
keep the older same-version metadata. `--no-index` works when all required
dependencies are installed or available in the wheel directory. The installed
validation environment needed the newly merged core's `zstandard` dependency;
its binary wheel was staged locally before resolving the extra.

## Source and results

The installed CLI was run outside the source worktree in the isolated provider
validation environment. First generation downloaded both pinned archives and
LICENSE into a fresh `ANYSEARCH_GAZEBO_CACHE`; pip installation downloaded no
assets. Offline fine generation reused that cache and read original SDF again.
No source checkout, native Gazebo, mesh renderer or C++ compiler is needed.

LICENSE SHA-256: `be970277b34c9fffa57b5a08d718bfc6a981308258be6b6f43badafc48444d32`.
Archive/revision hashes are in the governing spec and manifest constants.
The converter resolves 39 source collision shapes and 17 XML members.
BSD repository notice is retained, but reviewed use remains evaluation only;
the provider does not extend the older audit into a training/redistribution grant.

| Installed run | Extent | World identity |
| --- | --- | --- |
| Seed 42, 0.5 m | 184 x 18 x 184 | `875cfc89373e6bedba3cbac5edc0c97c6392e706e3393d5d3876f0f453da91d8` |
| Seed 42, 0.25 m, offline | 368 x 36 x 368 | `b9d282140017e732b9c8bdd7e8b830f7a5eca3eac4df9d47971007774ad71f62` |

Both runs verified three derived tasks, rejected the disconnected `west_to_east`
query and rendered six PNGs. Two accepted queries require altitude changes;
the third is a local planar control. Both roof checks found zero body-valid
cells over main-wall columns. The 0.5 m XY PNG was visually inspected.
The shared loader independently replays every witness against voxel cubes;
the exporter separately replays against source primitives and roof.

Corrected ignored outputs: `runtime/output/gazebo477-corrected-050` and
`runtime/output/gazebo477-corrected-025`; previews live under each `previews/`.
Source cache and wheels also remain under ignored `runtime/output` directories.
Original open-top world/conversion/occupancy and original split are preserved
as supplemental artifacts. Provider converter `2-provider` has its own immutable
identity and a self-contained roofed-world split; no v1 evidence is overwritten.
Seed rotates accepted task ordering only; it never changes the geometry identity.

## Validation and limits

Corrected focused provider plus Gazebo adapter suite: 31 passed. The prior
full provider plus adapter suite had 59 passing tests (one existing Typer
deprecation warning). Offline fixtures cover cache download/reuse/tamper,
explicit 4 MiB archive limit versus unchanged 2 MiB default, unsafe/duplicate/
oversized archive members, SDF transform composition, independent collision
rasterization, two resolutions, six PNGs, rights rejection, zero tasks,
CLI help without downloads and extra/entry-point isolation.

Installed `--help` prints correct layout/options but exits with an uncaught
external Click `Exit(0)` under this environment's newer bundled-Click Typer.
This pre-existing shared provider-group compatibility defect is tracked in
theseo-anysearch#471; child-command help tests do not cover the root exception
boundary. Generation and verification succeed. Do not report installed help
as clean until #471 is fixed.

The earlier `184 x 184 x 18` and `368 x 368 x 36` packs were invalid because
they mapped Gazebo Z-up onto renderer depth. They are superseded by #477 and
excluded from evidence. Correct packs map storage axes to source `(x,z,y)`.

`worlds add` is currently training-only. This evaluation/test-split provider
must reject training attachment without modifying the YAML; successful training
attachment requires a separately audited rights/version revision. Static maze
queries are not native simulator episodes, sensor observations, continuously
controlled flight trajectories, optimality proofs or a broad training corpus.
No training, GPU run, PR merge or PyPI publication was performed.
