# Resource benchmark

Use the adaptive resource benchmark to measure how many vectorized environments
and rollout workers a training workload can use efficiently on the current
machine. The initial implementation supports PPO, whose rollout controls are
fully propagated through the project trainer.

```bash
anysearch benchmark resources usage/experiments/train/ppo_baseline.yaml --open
```

The benchmark runs two phases:

1. It holds the rollout-worker count at one and increases environments per
   worker until sampled steps per second declines for the configured number of
   consecutive candidates.
2. It fixes environments per worker at the first phase's measured peak, then
   increases rollout workers until the median of the repeat-average GPU
   utilization reaches the configured target. The highest-throughput worker
   candidate measured before saturation is recommended.

Startup and actor-construction time are excluded. Each candidate is warmed up,
measured repeatedly, and compared using median end-to-end sampled steps per
second. A decline is counted only when throughput falls below the running best
by more than `--decline-tolerance`.

```bash
anysearch benchmark resources CONFIG \
  --decline-patience 3 \
  --decline-tolerance 0.02 \
  --warmup-iterations 1 \
  --measure-iterations 3 \
  --repeats 3 \
  --max-envs-per-worker 16 \
   --max-workers 20 \
   --max-gpu-utilization 95 \
   --max-duration-minutes 30 \
   --open
```

By default, the CLI replaces PPO and Ray startup chatter with one progress bar
per phase. Each completed tick reports throughput and average GPU utilization.
The first completed candidate creates `report.html`; `--open` opens it at that
point, and the page refreshes every five seconds as later candidates complete.
Use `--debug` to stream the captured PPO and Ray diagnostics to the terminal as
well as retaining them in the artifact directory.

Always set finite maximums. Candidate algorithms and Ray actors are stopped
after every repetition and Ray is shut down when the benchmark started it.
Rollout inference remains on CPU through `num_gpus_per_env_runner: 0.0`.
If GPU telemetry is unavailable or the target is not reached, worker scaling
continues through `--max-workers` and the report records that hard-limit stop.
The wall-clock budget is checked before each new candidate. An active RLlib
candidate finishes cleanly, and each phase always measures at least one
candidate, so total runtime can exceed the soft budget by those in-progress
measurements.

## Outputs

The command writes a timestamped directory below the experiment output path:

| Artifact | Contents |
|---|---|
| `report.html` | Standalone interactive Plotly report with throughput, speedup, GPU utilization, selected peaks, and stopping points. |
| `results.json` | Machine metadata, raw samples, median candidate summaries, stop reasons, and recommendation. |
| `results.csv` | One row per measured repetition for external analysis. |
| `recommended.yaml` | Recommended rollout settings ready to merge into a training config. |
| `benchmark.stdout.log` | Captured PPO stages and other standard output. |
| `benchmark.stderr.log` | Captured Ray, warning, and error output. |

The HTML report links directly to both diagnostic logs. Candidate-specific
runtime logs remain under `runs/<phase>/<candidate>/<repeat>/`, including each
repeat's `debug_stage.log` and Ray/RLlib runtime artifacts. If a candidate
fails, the CLI prints the relevant candidate log path and its stage log before
exiting.

When MLflow is enabled in the experiment, candidate summaries are logged as
metrics and all six top-level report files are logged as artifacts. GPU telemetry is
collected every 100 ms during measured iterations, reported as average
utilization across that region, and degrades gracefully when `nvidia-smi` is
unavailable.

The recommendation optimizes completed sampled steps per second, not GPU
utilization alone. A busier GPU is not an improvement when iteration throughput
declines.
