# Tune lifecycle and trial budgets

Tune trials report once after every completed training iteration. ASHA, PBT, and
other schedulers therefore see iterations 1 through N while the trial is still
running.

Each report contains reward and success metrics, total sampled environment
steps, and compute metadata:

- train batch size and PPO SGD iteration count;
- policy parameter count and flattened observation size;
- hidden-layer count and width;
- environment-runner and GPU allocation.

By default, every report includes a Ray checkpoint and every Tune iteration
writes a trajectory. This guarantees that an ASHA-pruned trial retains its
resolved experiment YAML, sampled configuration, metric history, latest policy
checkpoint, and replay artifact.

## Configuration

The tune_config block supports:

- checkpoint_frequency: report a restorable checkpoint every N iterations;
- preserve_trial_artifacts: force per-iteration trajectories and best replay;
- max_environment_steps: stop after the sampled-step budget;
- max_wall_time_s: stop after the wall-clock budget;
- target_success_rate: stop after deterministic evaluation reaches the target;
- metric: use evaluation_success_rate to optimize goal completion directly.

training.iterations remains the hard upper iteration budget.

## Resume behavior

Tune resume restores Ray's experiment state. When Ray restarts a function
trainable from a reported checkpoint, the worker restores the project Trainer
state and RLlib policy before continuing at the next iteration.

Each project trial directory contains experiment.yaml, sampled_config.yaml,
resources.json, tune_history.jsonl, tune_status.json, checkpoints, and
trajectories. A trial finishing below the maximum iteration budget is classified
as PRUNED unless it has an error, in which case it is FAILED.
