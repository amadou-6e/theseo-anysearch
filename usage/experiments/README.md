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
