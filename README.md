# Theseo AnySearch

Theseo AnySearch is a Python/Rust framework for navigation experiments in voxel
and surface environments. Compare classical pathfinding with reinforcement
learning, train and tune policies with Ray RLlib, and inspect saved trajectories
with native replay and policy-explanation tools.

The project supports single-agent Gymnasium and multi-agent PettingZoo
environments, imitation pretraining, staged curricula, and custom Python or
Rust experiment extensions. MLflow and TensorBoard record experiment results.

## Set up a development environment

Run the commands below from the repository root. The examples use PowerShell;
on Linux or macOS, activate the environment with `source .venv/bin/activate`.

Prerequisites:

- Python 3.10 or newer. GitHub Actions validates Python 3.12 on Windows.
- A stable Rust toolchain with Cargo, plus a native linker. On Windows, install
  Visual Studio Build Tools with the **Desktop development with C++** workload
  and a Windows SDK.
- Git and enough disk space for PyTorch, Ray, and the native build artifacts.

```powershell
git clone https://github.com/amadou-6e/theseo-anysearch.git
cd theseo-anysearch
git switch develop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable ".[dev,torch-cpu]" maturin
python -m maturin develop --release --manifest-path theseo_anysearch/core/Cargo.toml
anysearch --help
```

Install both packages: the root project provides the `anysearch` CLI, while
Maturin builds and installs the `theseo_core` native bindings into the active
virtual environment. Rebuild the bindings after changing Rust code.

The `torch-cpu` and `torch-gpu` extras currently both declare `torch>=2.0`; they
do not select a CPU-only or CUDA wheel. For a specific accelerator, install the
appropriate PyTorch build for your machine. Training can require a GPU with
`training.require_gpu: true`; the first example below needs no GPU.

## Run a first experiment

Start with the checked-in [Dijkstra baseline](usage/experiments/heuristics/dijkstra/README.md).
It searches a voxel action graph and writes a replayable trajectory without
training a neural network.

```powershell
anysearch run usage/experiments/heuristics/dijkstra/run.yaml
```

Run this from the repository root so that the configuration's
`usage/geometries/cube.stl` path resolves. The geometry is included in the
repository; no map download is needed. The example uses local tracking
(`mlflow: {}`), so a separate MLflow server is not required. Run artifacts are
written beneath `runtime/experiments/dijkstra/`.

Use the run identifier printed by the command to replay the result:

```powershell
anysearch replay dijkstra:<run-id>
```

Replay opens a native graphical interface and requires a desktop session.
For a small learning run, see the five-iteration
[PPO quick demo](usage/experiments/showcase/quick_demo.yaml) and the
[showcase guide](usage/experiments/showcase/README.md).

## Find your way around

| Directory | Responsibility |
| --- | --- |
| `theseo_anysearch/cli/` | Commands for running, tuning, inspecting, and replaying experiments |
| `theseo_anysearch/settings/` and `experiments/` | Typed configuration, run lifecycle, tracking, and artifacts |
| `theseo_anysearch/environments/` | Gymnasium and PettingZoo wrappers around native environments |
| `theseo_anysearch/core/` | Rust simulation, geometry, rendering, and PyO3 bindings |
| `theseo_anysearch/rllib/` | Policies, trainers, evaluation, curricula, tuning, and explanations |
| `theseo_anysearch/heuristic/`, `imitation/`, and `garden/` | Pathfinding baselines, demonstration pretraining, and pretrained encoders |
| `theseo_anysearch/extension_sdk/` | Rust SDK for experiment extensions |
| `usage/`, `docs/`, and `tests/` | Examples, detailed guides, and automated tests |

## Documentation

- [Documentation overview](docs/index.md)
- [Usage and examples](usage/README.md) and [experiment configuration](usage/experiments/README.md)
- [Settings](theseo_anysearch/settings/README.md), [geometry](docs/geometries/README.md),
  [actions](docs/actions/README.md), [observations](docs/observations/README.md),
  and [rewards](docs/rewards/README.md)
- [Training](docs/training/README.md), [staged training](docs/training/staging.md),
  [evaluation](docs/training/evaluation.md), and [tuning](docs/tuning/README.md)
- [Imitation pretraining](docs/imitation-pretraining.md) and
  [policy explanations](docs/explainability/README.md)
- [Native Rust extensions](docs/extensions/native_rust.md)

## Test and contribute

With the development dependencies and native bindings installed, run the local
suite from the repository root:

```powershell
python -m pytest -m "not ray and not integration" -q
```

See the [testing guide](docs/testing/README.md) for the marker contract and
commands for integration and real-Ray tests. These require additional runtime
resources and are selected separately from the lightweight suite.

Start changes with a GitHub issue, use a separate worktree and an issue branch
such as `fix/123` or `feat/123` from the latest `origin/develop`, and open a draft
pull request targeting `develop`. Keep generated runtime artifacts out of
commits. See [How to contribute](docs/how-to-contribute.md) for more detail.
