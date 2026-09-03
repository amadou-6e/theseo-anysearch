# Compiled world: valid vs. blocked

```powershell
python usage/experiments/showcase/geometry_capabilities/03_compiled_world/prepare_world.py
anysearch geometry validate usage/experiments/showcase/geometry_capabilities/03_compiled_world/experiment_valid.yaml --json
anysearch geometry validate usage/experiments/showcase/geometry_capabilities/03_compiled_world/experiment_blocked.yaml --json
```

`prepare_world.py` compiles two content-addressed 128x96x64 packs sharing one
extent, one start/goal pair, and one dividing wall at x=64-65 -- the *valid*
pack leaves a gap in the wall (y=41-56), the *blocked* pack fills it in. Both
`anysearch geometry` calls above route through `GeometryArtifact.bounded_reader`
(`theseo_anysearch/worlds/artifacts.py`), which decodes at most a handful of
32^3 chunks per query rather than materializing the pack, and enforces a hard
query budget.

Verified results:

| | exit code | `task_feasibility.rejection_reason` | `training_suitability.reason` |
|---|---|---|---|
| `experiment_valid.yaml` | `0` | `null` (108-step path found) | `null` |
| `experiment_blocked.yaml` | `1` | `no_path` | `no_path` |

That is the validation behavior actually distinguishing a solvable
large-world task from a broken one -- not a canned example.

## Replay the valid route

```powershell
python usage/experiments/showcase/geometry_capabilities/03_compiled_world/replay_valid_route.py
anysearch replay file runtime/geometry_capabilities/03_compiled_world/replay/trajectories/heuristic_astar.json
```

Walks the *valid* pack's 108-step solved route with the A* heuristic oracle
and writes a replayer-compatible trajectory (the *blocked* pack has no route
for the oracle to walk).

## Sparse-transformation bonus (labeled, not CLI-exposed)

Sparse compiled-world transformations (`theseo_anysearch/worlds/transformations.py`)
are not yet wired to any experiment YAML field or `anysearch geometry`
command -- there is no config or flag that installs a `SparseBoxTransform`
today. `prepare_world.py`'s last section calls that programmatic API directly
against the *valid* pack, layering a second, independently seeded wall over
its already-compiled occupancy entirely in memory, without touching the pack
on disk, and prints one point's occupancy with and without the overlay. That
call is demonstration glue you would write yourself today, not something
`anysearch geometry` runs for you.
