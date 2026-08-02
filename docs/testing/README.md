# Testing

The automated pull-request workflow runs the Local Test Suite, which does not
start or require a Ray runtime:

```powershell
python -m pytest -m "not ray and not integration" -q
```

This suite includes unit tests and other lightweight tests that may import or
mock Ray APIs. It excludes tests marked `ray`, which start or require a real Ray
runtime, and broader tests marked `integration`.

## Marker contract

- `ray`: starts or requires a real Ray runtime, including RLlib training and
  Tune sweeps.
- `integration`: exercises multiple runtime components or external processes.

Any test that calls `ray.init`, uses the `ray_session` fixture, launches an
RLlib algorithm, or executes a Tune sweep must carry the `ray` marker. A whole
module can declare it with:

```python
pytestmark = [pytest.mark.integration, pytest.mark.ray]
```

## Running excluded tests

Run the real-Ray suite explicitly on a machine with the required resources:

```powershell
python -m pytest -m ray -q
```

Run all integration tests, including integrations that do not require Ray, with:

```powershell
python -m pytest -m integration -q
```

The GitHub Actions workflow intentionally does not run either command.
## Automated job structure

The workflow validates and builds before running five test groups in parallel:

1. `Validate test contract` compiles Python and checks marker registration.
2. `Build Rust bindings` checks all Rust targets and uploads one wheel artifact.
3. Core/CLI, environment/heuristic, experiment/benchmarking, RLlib, and garden
   jobs download that same wheel and run their local tests independently.
4. `Local Test Suite` provides one aggregate result suitable for branch
   protection.

The parallel groups are path-disjoint, so each selected test runs once.

## Ray suite timing

The first dated Ray-suite timing baseline is recorded in
[`ray-suite-timings-2026-08-02.md`](ray-suite-timings-2026-08-02.md). It includes
the reproduction command, machine details, per-module durations, and known
fixture failures that currently make several measurements lower bounds.
