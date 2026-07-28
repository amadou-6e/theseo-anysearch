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
| `num_env_runners` | `0` | Dedicated RLlib evaluation workers; zero evaluates serially in the driver. |

## Evaluation-driven early stopping

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
  num_env_runners: 8
```

`training.num_env_runners` controls training sampling.
`evaluation.num_env_runners` controls deterministic evaluation separately. Evaluation episodes are assigned
round-robin to the dedicated workers, which receive synchronized policy weights
before every batch. Returned trajectories are sorted by seed before metrics,
early stopping, and replay artifacts are produced, so worker completion order
does not affect results.

A value of `0` preserves inline serial evaluation for constrained machines and
tests. Tune placement groups and local Ray CPU allocation reserve resources for
both training and evaluation workers. Evaluation does not overlap training.
