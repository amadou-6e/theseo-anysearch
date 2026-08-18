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
contains the resolved seed, episode index and scope, grid and occupied voxels,
action mode and offsets, prior scenario metadata, curriculum state, and YAML
parameters. Results contain a start plus either one goal or an ordered route,
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
