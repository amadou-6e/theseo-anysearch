# Quickstart

## Install

AnySearch is a Python 3.12 + Rust (PyO3/maturin) hybrid package. Install
it in editable mode with the dependency group matching your hardware:

```bash
pip install -e ".[torch-cpu]"   # or .[torch-gpu] on a CUDA machine
```

This registers the `anysearch` command.

## Run a training experiment

Every experiment is a single YAML file under `usage/experiments/`. Run
one directly by path:

```bash
anysearch run usage/experiments/train/ppo_baseline.yaml
```

Once a config has been run at least once, AnySearch auto-registers it
under its config filename, so subsequent runs can use the short form:

```bash
anysearch run ppo_baseline
```

For a hyperparameter sweep, point `run` at a `tune_config`-bearing YAML
instead — the same command dispatches to Ray Tune automatically:

```bash
anysearch run usage/experiments/tune/ppo_asha.yaml --tag my-sweep
```

## Inspect and resume runs

Every run prints a `run_id` on completion. Use it with `inspect` to
print the resolved config, metrics, and artifact paths:

```bash
anysearch inspect ppo_baseline:a1b2c3d4
```

Continue an interrupted run from its latest checkpoint with `resume`,
using the same `<name>:<run_id>` reference:

```bash
anysearch resume ppo_baseline:a1b2c3d4
```

## Track with MLflow

Training configs write to MLflow. By default this uses a local
`./mlruns` directory; point `mlflow.tracking_uri` in your experiment
YAML at a server to centralize tracking, e.g.:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Next steps

- Browse the {doc}`example geometries <geometries/README>` shipped with the repo
- See {doc}`training/README` for checkpointing, staging, and evaluation
- See {doc}`tuning/README` for Ray Tune sweep configuration
