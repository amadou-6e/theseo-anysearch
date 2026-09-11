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
3. Core/CLI, environment/heuristic, experiment/benchmarking, RLlib/imitation, and garden
   jobs download that same wheel and run their local tests independently.
4. `Local Test Suite` provides one aggregate result suitable for branch
   protection.

The parallel groups are path-disjoint, so each selected test runs once.

The RLlib/imitation job includes `tests/test_imitation` with the same
`not ray and not integration` marker filter. Its result is required by the
aggregate `Local Test Suite` gate. To run just these imitation tests locally:

```powershell
python -m pytest tests/test_imitation -m "not ray and not integration" -q
```

## Ray suite timing

The first dated Ray-suite timing baseline is recorded in
[`ray-suite-timings-2026-08-02.md`](ray-suite-timings-2026-08-02.md). It includes
the reproduction command, machine details, per-module durations, and known
fixture failures that currently make several measurements lower bounds.

The separate `Ray CLI Test Suite` GitHub Actions workflow runs the five modern
end-to-end CLI tests on pull requests to `develop`. It uploads JUnit timing data
as the `ray-cli-test-timings` artifact even when a test fails.
