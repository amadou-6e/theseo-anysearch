# Voxel architecture

The `voxel` module owns the complete voxel environment family. This includes its
environment state and lifecycle as well as every behavior contract that depends on
integer voxel coordinates, filled cells, action offsets, collision state, trails, or
voxel observations.

## Layout

```text
voxel/
|-- environment/
|   |-- geometry.rs        # Shared surface-cell and distance helpers
|   |-- multi.rs           # Multi-agent voxel environment
|   `-- single/
|       |-- mod.rs         # Single-agent state and configuration
|       |-- action_pipeline.rs # Predicate checks, action execution, and masking
|       |-- lifecycle.rs   # Reset, step, reward, and termination lifecycle
|       `-- tests.rs       # Single-agent behavioral tests
|-- actions/               # Shared offsets, history, and predicate/outcome pipeline state
|-- world/                 # Sparse voxel state, blocks, STL parsing, and voxelization
|-- sampling/              # Reusable STL-to-voxel geometry sampler
|-- rendering/             # Voxel episode traces and training-video generation
|-- common/                # Library, ABI version, name, and parameter validation
|-- predicates/            # Predicate ABI, native loader, context, and built-ins
|-- outcomes/              # Outcome ABI, native loader, context, and built-ins
|-- rewards/               # Reward ABI, configuration, breakdown, and built-ins
`-- metrics/               # Training/evaluation metric ABI, context, and result
```

`VoxelEnv` owns single-agent environment state and step ordering. It delegates action
feasibility to the predicate pipeline, successful action effects to the outcome pipeline,
and custom reward execution to the reward module. `MultiAgentVoxelEnv` shares the voxel
geometry helpers and reward configuration without living in the generic environment module.

`core/src/environments` now contains only environment-family-neutral traits and the distinct
surface environment. New voxel functionality belongs under this module, not under
`environments`.

Training and evaluation metrics are voxel-owned data contracts but are not executed during
`VoxelEnv::step`. The experiment/RLlib layer determines when a training or evaluation result
exists and supplies its serialized context. This keeps trainer scheduling out of the environment
while retaining a Rust implementation of the native metric ABI.

## Public API

The crate exports the voxel family from `crate::voxel`:

- `VoxelEnv`, `VoxelAction`, and `VoxelObservation`;
- `MultiAgentVoxelEnv`, `AgentEntry`, and `MultiStepResult`;
- RewardConfig, DistanceRewardMode, and ZoneRewardCurve;
- world storage and geometry-ingestion APIs;
- sampling::VoxelSampler.

Python bridge modules import these public voxel exports. Internal environment files are not
compatibility shims and are not re-exported through `crate::environments`.

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
|-- environment/
|-- predicates/
|-- outcomes/
|-- rewards/
`-- metrics/
```

Only infrastructure proven to be environment-independent across multiple families should move
to a shared module.