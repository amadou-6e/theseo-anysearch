# Scenario providers

Scenario providers select an episode's start and goal route immediately before
the Rust environment resets. They are useful when fixed waypoints and built-in
curriculum sampling cannot express a task distribution.

```yaml
env:
  scenarios:
    provider:
      name: adjacent_goal
      parameters:
        center: [16, 16, 16]

evaluation:
  scenarios:
    provider: adjacent_goal
```

Place Python providers in `scenarios.py` beside `experiment.yaml`. A provider
accepts one `ScenarioContext` and returns a `ScenarioResult`. The context
contains the resolved seed, episode index and scope, three-axis extent, world
identity, lazy continuous-grid `world` and `candidates` handles, action mode and
offsets, prior scenario metadata, curriculum state, and YAML parameters. It
never serializes the complete geometry. Results contain a start plus either one goal or an ordered route,
a stable scenario ID, and optional metadata.

Rust extensions use the same YAML name:

```rust
use anysearch_extension::{anysearch_scenario, ScenarioContext, ScenarioResult};

#[anysearch_scenario]
pub fn adjacent_goal(context: &ScenarioContext) -> ScenarioResult {
    ScenarioResult::goal([16, 16, 16], [17, 16, 16],
                         format!("adjacent-{}", context.episode_index))
}
```

Include the scenario capability bit (`32`) in
`anysearch_extension_capabilities`, then run `anysearch compile <experiment>`.
AnySearch archives the selected Python source or compiled native artifact with
the run. Missing functions, provider errors, invalid results, occupied cells,
and out-of-grid coordinates fail explicitly; no fallback scenario is used.

Python and Rust providers may coexist in the same experiment. Names exported
only by `scenarios.py` run in Python, while names exported by the native
manifest run in Rust. If both implementations expose the identical selected
name, Rust supersedes Python. Tune copies `scenarios.py` into every trial before
constructing its environment runners.

Evaluation uses its own provider block. Evaluation seeds are deterministic, so
a finite provider can map seeds to a fixed suite while retaining vectorized
evaluation and stable result ordering. The episode index remains useful for
single-environment providers but is local to each vectorized environment.

## Native world-query ABI v2

Use `#[anysearch_scenario_v2]` when a native provider needs world data. V2 is
additive: the v1 symbol, JSON context, fixed buffer behavior, and compatibility
policy are unchanged. The host invokes v2 below Python and supplies a
`#[repr(C)] WorldQueryApiV1`; no Rust reference, slice, trait object, closure, or
unstably represented enum crosses the dynamic-library boundary.

```rust
use anysearch_extension::{anysearch_scenario_v2, ScenarioContextV2, ScenarioResult};

#[anysearch_scenario_v2]
fn clear_line(context: &ScenarioContextV2<'_>) -> ScenarioResult {
    let obstacle = context.world.ray([1, 1, 1], [1, 0, 0], 30).unwrap();
    let end = obstacle.map_or(31, |hit| hit.coordinate.x.saturating_sub(1));
    ScenarioResult::goal([1, 1, 1], [end as i32, 1, 1], "clear-line")
}
```

The safe SDK exposes `point`, `region`, `ray`, and `count`. Coordinates are
zero-based storage coordinates. Region minima are inclusive and maxima are
exclusive. Extents must be non-empty, ordered, and within the logical world.
Ray steps are components in `-1..=1`, excluding `[0, 0, 0]`. Cold queries may
synchronously load through the world residency layer, but extensions cannot
observe chunks, pack offsets, cache state, or other residency details.

The callback table has its own ABI version and byte-sized structure length,
independent of scenario ABI v2. Consumers reject unknown versions and
undersized structures. Every pointer/length pair must be null with length zero
or valid for its declared element type and full length. Non-zero output lengths
require non-null writable pointers. Length multiplication is checked before
access. Region output is capped at 1,000,000 entries and scenario JSON at 1 MiB.

Region and scenario outputs use exact two-call negotiation: call with no
storage, read the required element count, allocate within the documented cap,
then call again. A short buffer returns `InsufficientBuffer` and the exact
required length. Empty/miss, block hit, invalid argument, out of bounds,
insufficient buffer, stale token, backend failure, unsupported operation, and
host panic/internal failure are distinct statuses. A backend error is never
converted into an empty voxel or ray miss.

The opaque context and call token, callback table, input strings, and output
buffers are valid only on the invoking thread and only until the scenario call
returns. Extensions must not retain them. Calls from another thread, nested
scenario invocation, callback reentry, and stale tokens are rejected. Both
host callbacks and the SDK export catch panics so unwinding never crosses FFI.
See `usage/experiments/showcase/scenario_world_query_v2` for a complete example.

## Candidate-index migration

Compiled worlds store `candidates.idx` and `candidates.bin` beside the world
pack. Records use public one-based coordinates and are bucketed by coarse
region and kind (`spawn`, `goal`, `surface`, or `portal`). Configure the
compiled-world directory under `env.scenarios.candidate_index`; every reset
receives a fresh lazy handle with independent query and result budgets.

Python providers previously reading `context.grid_size` should use
`context.extent`. Providers previously scanning `context.filled_voxels` should
use bounded `context.world` queries or `context.candidates.sample(...)` with an
explicit `seed` and `stream`. Candidate results may be filtered by kind,
region, distance, radius, and minimum quality. Identical world identity, seed,
stream, and filters produce identical results across processes and cache
states. Providers must not depend on file order or query history.
