# Checkpoint execution recipes

Issue #473 introduces versioned recipes for re-executing a saved policy without
silently inheriting current defaults. Clone into a portable bundle, inspect the
effective configuration, then explicitly select the execution scope:

```powershell
anysearch clone --checkpoint <run>/checkpoints/iter_000100 --scope evaluation --bundle-dir <bundle> --output recipe.yaml
anysearch apply <bundle>/recipe.yaml --dry-run
anysearch apply <bundle>/recipe.yaml --output-dir <evaluations> --episodes 10 --seed 42
```

The recipe records the complete checkpoint-tree hash, resolved experiment hash,
policy-facing action/observation/task/reward settings, archived extension binary
and manifest hashes, extension capability bindings, checkpoint state and known
provenance gaps. Validation fails on missing or changed artifacts. World changes,
scope-inactive state and unresolved provenance are printed explicitly.
Archived YAML is explicitly migrated to the current typed schema; the migration
and materialized defaults are recorded and validated before execution. A world
replacement also requires explicit `--task` and `--routes` decisions (`preserve`,
`clear`, or `replace`) and may be previewed with `--dry-run`.

Evaluation makes learner, optimizer, exploration and curriculum adaptation
inactive. It saves a fresh run ID, all requested trajectories, metrics, the
effective/runtime configuration, and before/after policy and checkpoint hashes.
Disabling an
extension binding requires both an existing qualified binding such as
`reward:segment_countdown_goal` and an explicit replacement.

Continuation restores the full checkpoint, including RLlib learner and optimizer
state and iteration/episode counters. It refuses a changed world or capability,
or a checkpoint lacking required checkpoint-local curriculum/early-stop state.
Fine-tuning starts a new optimizer, counters, curriculum and early-stop state,
but transfers and verifies the source policy weights. Both modes require an
explicit additional/target iteration limit:

```powershell
anysearch clone --checkpoint <run>/checkpoints/iter_000100 --scope continuation --bundle-dir <bundle> --output recipe.yaml
anysearch apply <bundle>/recipe.yaml --output-dir <runs> --iterations 10
```

For continuation, `--iterations 10` means at most ten *additional* iterations;
for fine-tuning it means ten iterations in the new run. The executor currently
supports single-agent PPO and bundled compiled geometry or a generated grid.
Historical checkpoints without per-checkpoint auxiliary state can still be
evaluated or fine-tuned, but cannot claim exact continuation.
