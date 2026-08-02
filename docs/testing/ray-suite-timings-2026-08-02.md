# Ray suite timing baseline — 2026-08-02

This baseline records one serial attempt of every test marked `ray`. It is an
execution measurement and a failure inventory, not a clean performance result:
many tests did not reach their intended training workload because of existing
Windows fixture and permission errors.

## Environment

- OS: Windows
- CPU: Intel Core i7-13850HX, 20 cores / 28 logical processors
- GPU: NVIDIA RTX A1000 6 GB Laptop GPU
- Python: 3.12.13
- Ray: 2.56.1
- Torch: 2.12.1+cu130
- Pytest: 9.1.1
- Execution: one serial Pytest process; no Ray cluster was active at start

## Reproduction

```powershell
python -m pytest -m ray -q --durations=0 `
  --junitxml=runtime/test_timings/ray-suite.xml
```

The JUnit XML contains the duration and outcome of all 145 collected Ray tests.
The `runtime/` copy is intentionally not committed; this dated report preserves
the relevant aggregate measurements.

## Result

| Measurement | Value |
|---|---:|
| Collected Ray tests | 145 |
| Wall-clock duration | 1,496.348 s (24m 56.3s) |
| Sum of testcase durations | 1,403.495 s (23m 23.5s) |
| Collection/session overhead | 92.853 s (1m 32.9s) |
| Passed | 31 |
| Failed | 48 |
| Setup errors | 54 |
| Skipped | 12 |

## Duration by area

| Area | Tests | Recorded duration |
|---|---:|---:|
| CLI integration | 36 | 1,050.630 s (17m 30.6s) |
| RLlib integration | 92 | 332.449 s (5m 32.4s) |
| Experiment integration | 17 | 20.416 s |

The experiment, CNN, DQN, and SAC totals are lower bounds because shared setup
failed before their intended training work ran.

## Module timings

| Module | Tests | Passed | Failed | Errors | Skipped | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| `test_cli_ray` | 13 | 3 | 10 | 0 | 0 | 669.624 |
| `test_cli_commands` | 23 | 4 | 7 | 0 | 12 | 381.006 |
| `test_tune_runner_ray` | 19 | 4 | 15 | 0 | 0 | 135.101 |
| `test_ppo_trainer_ray` | 16 | 16 | 0 | 0 | 0 | 104.637 |
| `test_algorithm_coverage` | 11 | 4 | 7 | 0 | 0 | 90.623 |
| `test_experiment_runner_ray` | 15 | 0 | 0 | 15 | 0 | 20.383 |
| `test_cnn_models_ray` | 16 | 0 | 6 | 10 | 0 | 1.518 |
| `test_dqn_trainer_ray` | 15 | 0 | 0 | 15 | 0 | 0.263 |
| `test_sac_trainer_ray` | 14 | 0 | 0 | 14 | 0 | 0.182 |
| `test_ray_ppo_real` | 1 | 0 | 1 | 0 | 0 | 0.125 |
| `test_train` | 2 | 0 | 2 | 0 | 0 | 0.033 |

## Dominant blockers

- 56 tests encountered a missing `\tmp\toy.stl`; Unix-style fixture paths are
  not converted into valid Windows temporary paths.
- 22 tests encountered `PermissionError: [WinError 5]`.
- Four tests encountered an undefined `_resolve_pool_dir` name.
- Additional CLI failures exposed stale fixtures, including
  `trajectory_every: 0`, which the current settings model rejects.

These failures mean 24m 56.3s is not the expected duration of a passing suite.
It is the measured duration of the current suite and should be treated as a
lower bound until the blockers are fixed and the benchmark is repeated.

## Initial CI partition proposal

Use separate jobs so independent Ray runtimes do not serialize all startup and
training work:

1. CLI Ray tests — initial timeout: 30 minutes.
2. Tune and algorithm coverage — initial timeout: 20 minutes.
3. Trainer and model integration — initial timeout: 30 minutes.
4. Experiment runner integration — initial timeout: 20 minutes.

Keep a 60-minute timeout only for an unsplit fallback job. Re-run this benchmark
after the fixture failures are corrected before tightening any timeout.
## CLI reduction follow-up

The deprecated `anysearch experiment ...` and `anysearch tune ...` Ray tests
were removed. Their compatibility and deprecation notices remain covered by CLI
unit tests. The modern CLI Ray tests were consolidated around one shared initial
run, leaving five end-to-end tests for run, inspect, resume, repeat, and list.

The reduced suite was measured with:

```powershell
python -m pytest tests/test_cli/test_integration/test_cli_commands.py `
  -m ray -q --durations=0 `
  --junitxml=runtime/test_timings/cli-ray-reduced-final.xml
```

| Measurement | Before | After |
|---|---:|---:|
| CLI Ray tests | 36 | 5 |
| CLI Ray wall time | 1,050.630 s | 199.229 s |
| Outcome | 7 passed, 17 failed, 12 skipped | 5 passed |
| Runtime reduction | — | 81.0% |

The five passing test durations were:

| Operation | Seconds |
|---|---:|
| Initial run and shared fixture | 77.80 |
| Repeat | 58.41 |
| Resume | 47.36 |
| List | 9.50 |
| Inspect | 4.54 |

This run also exposed and fixed archived-run output rebasing: resume and repeat
loaded `run_dir/experiment.yaml`, which incorrectly changed the output root to
the run directory. Both commands now restore the original root represented by
`<output>/<experiment>/<run_id>` before constructing `ExperimentRunner`.
