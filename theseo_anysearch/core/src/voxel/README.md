# Voxel behavior architecture

The `voxel` module owns behavior contracts that depend on voxel-specific state:
cursors, integer coordinates, filled cells, action offsets, collision state, trails,
and voxel observations. Other environment families should define their own predicates,
outcomes, rewards, and metrics instead of depending on these contracts.

## Layout

```text
voxel/
|-- actions/       # Action history and ordered predicate/outcome pipeline state
|-- common/        # Library, ABI version, name, and JSON parameter validation
|-- predicates/    # Predicate ABI, native loader, context, and built-ins
|-- outcomes/      # Outcome ABI, native loader, context, and built-ins
|-- rewards/       # Reward ABI, configuration, breakdown, loader, and built-ins
`-- metrics/       # Training/evaluation metric ABI, context, loader, and result
```

`VoxelEnv` owns environment state and step ordering. It delegates feasibility to the
predicate pipeline, successful action effects to the outcome pipeline, and custom reward
execution to the reward module. Built-in and native implementations share the same contexts.

Training and evaluation metrics are voxel-owned data contracts but are not executed during
`VoxelEnv::step`. The experiment/RLlib layer determines when a training or evaluation result
exists and supplies its serialized context. This keeps trainer scheduling out of the environment
while retaining a Rust implementation of the native metric ABI.

## Compatibility contract

Internal Rust module paths are not part of the native extension ABI. The following are stable:

- ABI version `2` for reward, predicate, and outcome libraries;
- field order and types of every `#[repr(C)]` structure;
- `anysearch_reward_<name>_v2`;
- `anysearch_predicate_<name>_v2`;
- `anysearch_outcome_<name>_v2`;
- `anysearch_compute_training_metrics_v1`;
- `anysearch_compute_evaluation_metrics_v1`;
- YAML selector names and extension capability flags.

The extension SDK mirrors these layouts. Changes to an ABI structure require a new ABI version;
moving or renaming an internal Rust module does not.

## Adding another environment family

A non-voxel environment should create its own ownership root, for example:

```text
continuous/
|-- predicates/
|-- outcomes/
|-- rewards/
`-- metrics/
```

Only infrastructure proven to be environment-independent across multiple families should move
to a shared runtime module.
