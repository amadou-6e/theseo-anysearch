# Experiment catalog

This directory contains runnable AnySearch experiment definitions. Launch a configuration with:

```powershell
anysearch run usage/experiments/<category>/<file>.yaml
```

Each YAML starts with a `description` field that is loaded as experiment metadata and explains its intended behavior.

## Categories

- [`heuristics/`](heuristics/README.md) contains standalone graph-search planners that do not train a neural policy.
- [`showcase/`](showcase/README.md) contains short demonstrations and pipeline smoke tests.
- [`train/`](train/README.md) contains longer baseline and task-specific policy-training configurations.
- [`tune/`](tune/README.md) contains hyperparameter searches and cross-geometry sweeps.

Runtime directories, checkpoints, trajectories, and copied `experiment.yaml` snapshots are generated artifacts rather than source configurations.
## Environment blocks

Environment representation is grouped by responsibility:

```yaml
env:
  seed: 42
  agent_count: 1
  max_steps: 96  # 32 + 32 + 32
  trail_mode: true
  geometry:
    stl_path: usage/geometries/stepped_terrain.stl
    scale: 40.0
    grid_size: 32
  observation:
    mode: box
    box_radius: 1
  action:
    mode: discrete_26
  rewards:
    step_cost: -0.01
    goal_reward: 1.0
    distance_shaping: 0.2
```

Legacy flattened environment fields remain loadable during migration. A file may not mix a nested block with legacy fields belonging to that same block.
