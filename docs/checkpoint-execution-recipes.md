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
`clear`, or `replace`) and may be previewed with `--dry-run`:

```powershell
anysearch apply <bundle>/recipe.yaml --world <compiled-world>/manifest.json --task preserve --routes preserve --dry-run
anysearch apply <bundle>/recipe.yaml --world <compiled-world>/manifest.json --task preserve --routes preserve --output-dir <evaluations> --episodes 10 --seed 42
```

Dry-run verifies artifact hashes, compiled-world integrity, and that the
effective action/observation spaces still fit the saved policy. It creates no
run. A recipe with `execution_supported: false` prints its specific blocker;
running it without `--dry-run` exits nonzero. Replacing routes or task content
requires `--routes-config` or `--task-config` only with the matching `replace`
decision. A route-mode change that alters policy observation shape is refused.

Evaluation makes learner, optimizer, exploration and curriculum adaptation
inactive. It saves a fresh run ID, all requested trajectories, metrics, the
effective/runtime configuration, and before/after policy and checkpoint hashes.
Disabling an extension binding requires both an existing qualified binding such
as `reward:segment_countdown_goal` and an explicit replacement, for example
`--disable-capability reward:segment_countdown_goal --replace-capability reward:segment_countdown_goal=reward:builtin`.
Multiple qualified replacements can be supplied in one invocation.

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
Staged-run checkpoints are rejected for training execution until stage state is
captured and restored at the checkpoint boundary.
Historical checkpoints without per-checkpoint auxiliary state can still be
evaluated or fine-tuned, but cannot claim exact continuation.

Each execution writes a new ID under `--output-dir`. `resolved_recipe.json` and
`difference_manifest.json` record what was preserved or changed; evaluation
adds `runtime_env_config.json`, `metrics.json`, `execution.json`, and one replay
per requested episode under `trajectories/`. Training writes
`effective_config.json`, a final checkpoint, and `execution.json` with the
source and initial policy hashes. Caught interruptions after output creation
write `execution_failure.json` instead of a success record; a hard process kill
can leave only partial files.
Outputs cannot be placed inside the bundle or any verified source artifact.
