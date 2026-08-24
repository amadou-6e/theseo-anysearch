# AnySearch

Rust-backed voxel reinforcement learning environments, driven by YAML
experiment configs and trained with Ray RLlib.

AnySearch is a Python 3.12 + Rust (PyO3/maturin) hybrid package for
training RL agents to navigate voxelised 3-D geometry — arbitrary STL
meshes voxelised at a configurable scale, stepped through Rust-backed
environments for fast throughput, with multi-agent support sharing one
environment instance.

## Features

**Environments**
- Rust-backed voxel environments (`PyVoxelEnv`, `PySurfaceEnv`) via PyO3 — fast step throughput
- STL geometry loading: arbitrary meshes voxelised at configurable scale
- Multi-agent support: N agents share one environment instance
- Observation modes: `scalar` (fill %, steps), `box` (3-D local grid), `radial` (ray casts), `hierarchical_box` (multi-resolution)
- Configurable reward shaping: step cost, goal reward, distance shaping, collision penalty
- Trail mode: cursor auto-fills visited cells (navigation task variant)

**Algorithms**
- PPO (full, legacy RLlib API stack — dict obs space compatible)
- SAC (full, continuous/discrete actions)
- DQN + Rainbow (full, prioritised experience replay, distributional Q)
- TD3, DDPG (stubs — require Box action space)
- All algorithms self-register; adding a new one requires only a single subclass

**Models**
- Fully-connected policy (configurable depth + width via `model_config`)
- `VoxelBox2DCNN` — Conv2d with z-as-channels for box observations
- `VoxelBox3DCNN` — Conv3d volumetric encoder for box observations
- Custom model passthrough: any RLlib-registered model via `custom_model`

**Training**
- YAML-driven experiment config (env + training + algorithm + model + MLflow in one file)
- Checkpointing at configurable intervals; resume from any checkpoint
- Trajectory recording: best episode always saved; periodic snapshots every N iterations
- Protobuf + JSON trajectory format for the eframe replay viewer

**Hyperparameter Search**
- Ray Tune integration: ASHA (early stopping), PBT (population-based training)
- YAML search space: all `ray.tune.*` samplers configurable without code changes
- Multi-geometry sweep: same config runs across multiple STL files
- MLflow tracking of all trials: parent run + nested child runs per trial

**Observability**
- MLflow tracking: params, per-iteration metrics, artifacts, run lifecycle
- `anysearch mlflow` launches the UI pointed at the right DB
- TensorBoard logs written automatically by Ray Tune to the session artifacts dir
- `anysearch inspect` prints resolved config + metrics summary for any run

**Infrastructure**
- Python 3.12 + Rust (PyO3/maturin) hybrid package
- GPU training via `require_gpu: true` in training config
- Anyscale cloud runner support (`runner: anyscale`)

```{toctree}
:maxdepth: 2
:caption: Getting started

quickstart
```

```{toctree}
:maxdepth: 1
:caption: Actions
:glob:

actions/*
```

```{toctree}
:maxdepth: 1
:caption: Curriculums
:glob:

curriculums/*
```

```{toctree}
:maxdepth: 1
:caption: Experiments
:glob:

experiments/*
```

```{toctree}
:maxdepth: 1
:caption: Explainability
:glob:

explainability/*
```

```{toctree}
:maxdepth: 1
:caption: Extensions
:glob:

extensions/*
```

```{toctree}
:maxdepth: 1
:caption: Geometries
:glob:

geometries/*
```

```{toctree}
:maxdepth: 1
:caption: Observations
:glob:

observations/*
```

```{toctree}
:maxdepth: 1
:caption: Rewards
:glob:

rewards/*
```

```{toctree}
:maxdepth: 1
:caption: Testing
:glob:

testing/*
```

```{toctree}
:maxdepth: 1
:caption: Training
:glob:

training/*
```

```{toctree}
:maxdepth: 1
:caption: Tuning
:glob:

tuning/*
```

```{toctree}
:maxdepth: 1
:caption: Reference

imitation-pretraining
task-contract
how-to-contribute
```
