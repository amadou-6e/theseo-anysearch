# Goal-finding evaluation

Every training iteration evaluates the current policy with exploration disabled
in fresh environments. The batch always starts at
evaluation.seed (default 42) and contains
evaluation.episodes consecutive seeds, so iterations are directly
comparable.

evaluation.min_success_rate (default 0.5) is the release threshold. A
policy is classified as solved only when the full evaluation batch meets that
threshold. A policy that improves goal distance but records zero successes is
classified as approaching_not_solved, never as solved.

CLI/run results, Tune, MLflow, TensorBoard, and per-iteration JSON summaries use
the same scalar metric map. It includes success count and rate, steps-to-goal
minimum/mean/maximum, final and minimum Euclidean goal distance,
terminated/truncated counts, shaped return, and reconstructed unshaped task
return.

## Fields

| Field | Default | Purpose |
|---|---:|---|
| `episodes` | `1` | Deterministic episodes collected after each training iteration. |
| `seed` | `42` | First seed in every deterministic evaluation batch. |
| `min_success_rate` | `0.5` | Success-rate threshold used to classify the policy as solved. |
| `num_env_runners` | `0` | Dedicated RLlib evaluation workers; zero evaluates in the driver. |
| `num_envs_per_env_runner` | `1` | Evaluation environments batched within each worker or the driver. |
| `parallel_to_training` | `false` | Let RLlib overlap deterministic evaluation with the training update. |

## Evaluation-driven early stopping

RLlib schedules every evaluation batch through the AnySearch custom evaluation
function, whether evaluation is sequential or parallel. The trainer does not
run a second, independent evaluation loop.

`training.early_stop` consumes these deterministic evaluation batches. Reward
and goal-finish conditions use the batch mean reward and completed-goal count.
Heuristic accuracy compares the policy and configured heuristic on the same
seed sequence. Training data and exploratory rollouts never trigger a stop.

## Parallel evaluation

Evaluation uses a dedicated RLlib EnvRunner pool when configured:

```yaml
training:
  num_env_runners: 8

evaluation:
  episodes: 20
  seed: 42
  min_success_rate: 0.5
  num_env_runners: 4
  num_envs_per_env_runner: 4
  parallel_to_training: true
```

`training.num_env_runners` controls training sampling.
`evaluation.num_env_runners` controls dedicated evaluation workers, while
`evaluation.num_envs_per_env_runner` controls the vector width within each worker.
Active observations are passed through one batched policy inference call. Dedicated
workers receive synchronized policy weights before every batch. Returned trajectories are sorted by seed before metrics,
early stopping, and replay artifacts are produced, so worker completion order
does not affect results.

A `num_env_runners` value of `0` evaluates inline and still supports vectorization,
but cannot be combined with `parallel_to_training: true`.
The effective concurrency is limited by the episode count and equals at most
`max(num_env_runners, 1) * num_envs_per_env_runner`. Tune placement groups and local Ray CPU allocation reserve resources for
both training and evaluation workers. When parallel evaluation is enabled,
RLlib schedules the AnySearch custom evaluator concurrently and guarantees that
its workers receive a synchronized policy snapshot. The reported evaluation is
one policy update behind the training metrics for that iteration; checkpoint
selection must account for that documented RLlib behavior. AnySearch still
processes the returned episodes through its normal metrics, trajectory, and
early-stopping pipeline.
