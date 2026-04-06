# Voxel Renderer Architecture Spec (PoC)

## Goals

1. Split the application into clear folders for `app`, `ui`, and `world`.
2. Define one stable world interface that supports `set`, `remove`, and `update` block operations.
3. Add a dedicated import pipeline for geometry ingestion (`.stl` first).
4. Add a dedicated environments module for reinforcement-learning transition/reward rules.
5. Prepare a Python boundary where CLI, geometry submission, environment config, and RL orchestration (including Anyscale workflows) can live.

## Folder Layout

```text
src/
  app/
    mod.rs
    main.rs
    ui/
      mod.rs
      panels.rs
  world/
    mod.rs
    api.rs
    block.rs
    state.rs
  import/
    mod.rs
    stl.rs
    voxelize.rs
  envs/
    mod.rs
    traits.rs
    basic_voxel_env.rs
  bridge/
    mod.rs
    dto.rs
    py_bindings.rs
  camera.rs
  renderer.rs
  lib.rs
  main.rs
  bin/
    poc.rs
```

## World Interface Contract

```rust
pub trait WorldInterface {
    fn set_block(&mut self, coord: Coord, block: Block) -> Result<(), WorldError>;
    fn remove_block(&mut self, coord: Coord) -> Result<(), WorldError>;
    fn update_block(&mut self, coord: Coord, update: BlockUpdate) -> Result<(), WorldError>;
    fn get_block(&self, coord: Coord) -> Option<&Block>;
}
```

### Behavioral Rules

1. `set_block` inserts or replaces a block at a coordinate.
2. `remove_block` returns `NotFound` when no block is present.
3. `update_block` mutates an existing block and returns `NotFound` if missing.
4. All operations validate bounds in `[0..WORLD_SIZE)` and return `OutOfBounds` when invalid.

## Import Pipeline (`import/`)

1. Parse geometry:
   1. `parse_ascii_stl(...)` converts STL text into a mesh model.
2. Voxelize:
   1. `voxelize_mesh(...)` projects mesh vertices into voxel coordinates.
3. Apply:
   1. Generated placements can be applied through `WorldInterface::set_block`.

PoC scope:
1. ASCII STL only.
2. Vertex-based occupancy (fast proof), not full watertight solid fill.

## Environments (`envs/`)

`RlEnvironment` defines `reset` and `step` for RL loops. A sample `BasicVoxelEnv` is included:

1. Action space:
   1. `Place(coord)`
   2. `Remove(coord)`
   3. `Noop`
2. Reward:
   1. +1 when filled voxel count reaches/gets closer to target.
   2. Small penalty for invalid operations.
3. Done:
   1. Target reached, or `max_steps` exhausted.

## Python Boundary (`bridge/`)

PoC boundary uses DTOs and orchestration functions in Rust:

1. Geometry submission DTO (`GeometrySubmission`)
2. Environment config DTO (`EnvConfig`)
3. Bridge function `run_submission_poc(...)` that:
   1. Parses + voxelizes STL.
   2. Applies placements to world through the interface.
   3. Runs a short environment rollout.
   4. Returns a compact summary suitable for a Python CLI.

Future production boundary:
1. Add `pyo3` wrappers in `bridge/py_bindings.rs`.
2. Build wheel with `maturin`.
3. Keep RL training/Anyscale orchestration in Python package while delegating simulation-heavy ops to Rust.

## PoC Acceptance Criteria

1. `cargo check` succeeds.
2. `cargo run --bin poc` succeeds.
3. PoC output demonstrates:
   1. World set/remove/update API calls.
   2. STL parse + voxel placement.
   3. Environment reset/step loop.
   4. Bridge-level orchestration summary.
