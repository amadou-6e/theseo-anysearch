# Checkpoint execution recipes

Issue #473 introduces versioned recipes for re-executing a saved policy without
silently inheriting current defaults. The initial safe slice supports cloning and
read-only validation:

```powershell
anysearch clone --checkpoint <run>/checkpoints/iter_000100 --scope evaluation --output recipe.yaml
anysearch apply recipe.yaml --world path/to/manifest.json --dry-run
```

The recipe records the complete checkpoint-tree hash, resolved experiment hash,
policy-facing action/observation/task/reward settings, archived extension binary
and manifest hashes, extension capability bindings, checkpoint state and known
provenance gaps. Validation fails on missing or changed artifacts. World changes,
scope-inactive state and unresolved provenance are printed explicitly.
Archived YAML is captured without forcing it through the current configuration
schema; schema migration belongs to apply-time compatibility validation.

Evaluation makes learner, optimizer, exploration and curriculum adaptation
inactive. Continuation and fine-tuning remain distinct scopes. Disabling an
extension binding requires both an existing qualified binding such as
`reward:segment_countdown_goal` and an explicit replacement.

Execution is deliberately blocked in this slice: `apply` without `--dry-run`
fails before loading the policy or creating output. The executor must next add
portable path resolution, task/world dependency validation, policy restoration,
evaluation output, continuation and explicit fine-tuning initialization. A dry
run is not evidence that these unimplemented operations are supported.
