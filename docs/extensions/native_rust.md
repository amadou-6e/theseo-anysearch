# Native Rust extensions

AnySearch experiments can define reward and metric hooks in Rust. Put a Cargo
`cdylib` crate in the experiment's `extension/` directory. Its `Cargo.toml` uses
the SDK version bundled with AnySearch:

```toml
[dependencies]
anysearch-extension = "0.1.0"
```

Compile it with:

```powershell
anysearch compile path/to/experiment
```

The command resolves the bundled SDK, generates `Cargo.lock`, builds a release
library, validates ABI version 2,
and stores a source-hashed artifact beneath the ignored `.anysearch/` directory.
A subsequent command reuses the artifact when its sources and binary hash still
match. Use `--force` to rebuild it.

At run startup, AnySearch rejects stale sources or an incompatible library. It
copies the exact manifest and platform library into the run directory; Tune does
the same independently for every trial. Worker processes load this archived copy.
Artifacts are platform-specific and should be recompiled on the target machine.

## ABI and precedence

Every library exports `anysearch_extension_abi_version` and
`anysearch_extension_capabilities`. It may then export any combination of:

- `anysearch_reward_<yaml-name>_v2`
- `anysearch_compute_training_metrics_v1`
- `anysearch_compute_evaluation_metrics_v1`

Rewards use fixed-layout C-compatible context/result structs because this call is
made on every environment step. A result contains add/replace mode, a finite
reward, and up to eight named components. The reward context includes both the
episode-wide `step` and the current route segment's `segment_step` and
`segment_length`; `segment_step` resets when a new waypoint becomes active.
Metrics receive UTF-8 JSON and return a
JSON object because metric calls are infrequent and benefit from a flexible
context. Returned names get the normal `training_` or `evaluation_` prefix.

A compiled Rust reward takes precedence over a Python reward with the selected
name. Python and Rust metrics both execute; a Rust metric supersedes only a Python
metric with the same scoped name. This makes partial migrations possible.

Native extensions are trusted code loaded into the training process. Only compile
and run extension sources you trust. See
`usage/experiments/showcase/native_extension` for a runnable example using the
`anysearch-extension` SDK. The `#[anysearch_reward]` macro generates the stable,
versioned ABI export; extension authors do not write that wrapper themselves.

## Action predicates and outcomes

The action pipeline runs inside the Rust environment. A behavior preset selects a complete
predicate/outcome set, while explicit lists allow an experiment to replace either side:

```yaml
env:
  action:
    mode: discrete_18
    behavior: trail_navigation
    history_length: 16
```

`cursor_navigation` checks that an action is valid, in bounds, and unoccupied, then
moves the cursor. `trail_navigation` uses the same predicates and also fills the
destination voxel. `legacy` preserves the old `trail_mode` behavior.

For a custom pipeline, selectors accept a name or a name plus JSON parameters:

```yaml
env:
  action:
    mode: discrete_18
    predicates:
      - valid_action
      - bounds
      - unoccupied
      - name: avoid_repeated_collision
        parameters: {}
    outcomes:
      - cursor_movement
      - name: mark_destination
        parameters: {}
    history_length: 8
```

Predicates receive the current environment-derived state, proposed action and destination,
observation scalars, and bounded action history. Every predicate must allow the action.
The same Rust predicate evaluation powers `env.action_mask()` and step feasibility.

To expose that feasibility mask to a discrete RLlib policy and apply it to the
policy logits, enable masking under the action configuration:

```yaml
env:
  action:
    mode: discrete_26
    masking:
      enabled: true
      all_masked: error
```

Masking is opt-in, so existing observations and policies are unchanged. The
mask is added to the observation as `action_mask`; voxel CNN models exclude it
from encoder features and set rejected-action logits to the minimum finite
value before sampling. If predicates reject every action, the environment
raises immediately because current discrete spaces do not expose a no-op.

`vector_3` uses factorized `MultiDiscrete` logits, which cannot exactly encode
an arbitrary joint 27-movement mask. Configuration validation therefore
rejects masking with `vector_3` instead of silently sampling forbidden joint
actions.
Outcomes run only after feasibility succeeds and return validated mutations such as moving
the cursor, placing a voxel, or removing an agent voxel. Mutations are applied atomically.

Custom Rust functions use normal names; the macros generate the versioned ABI exports:

```rust
#[anysearch_predicate]
pub fn avoid_repeated_collision(context: &PredicateContext) -> PredicateResult {
    PredicateResult::allow()
}

#[anysearch_outcome]
pub fn mark_destination(
    context: &OutcomeContext,
    mutations: &mut OutcomeMutations,
) -> OutcomeResult {
    mutations.place_voxel(context.destination);
    OutcomeResult::applied()
}
```

A custom export with the selected name supersedes the built-in of the same name. The
extension must advertise predicate bit `8` and outcome bit `16` in
`anysearch_extension_capabilities`. Compile it with `anysearch compile` as usual.
## Core ownership

The Rust core keeps voxel-specific extension contracts under
`theseo_anysearch/core/src/voxel/`:

- `predicates/` owns feasibility contexts, built-ins, and native loading;
- `outcomes/` owns validated post-action mutations and native loading;
- `rewards/` owns built-in reward configuration, named breakdowns, and native loading;
- `metrics/` owns the native metric function contract;
- `actions/` owns ordered predicate/outcome pipeline state;
- `common/` contains validation shared only by these voxel behaviors.

This placement is deliberate. Cursor coordinates, filled cells, trail placement, collision
state, and voxel observations are not universal environment concepts. A future surface,
continuous, or graph environment can define a different set of contracts without inheriting
the voxel API.

Internal module movement does not change compiled extensions. Compatibility is determined by
the exported symbol names, capability flags, ABI version, and `#[repr(C)]` layouts. Training
and evaluation metric scheduling remains in the experiment layer because only that layer owns
complete trainer and evaluator results; the voxel metric module owns the native invocation
contract, not evaluation scheduling.
