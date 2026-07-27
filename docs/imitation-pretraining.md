# Heuristic imitation pretraining

The imitation stage teaches the PPO policy to copy successful heuristic actions
before normal reinforcement learning begins:

```text
heuristic demonstrations -> behavior cloning -> PPO policy -> normal PPO
```

Enable it with a top-level `imitation` block. `training.algorithm` remains
`ppo`; the heuristic is only the teacher.

## Configuration

- `enabled` runs imitation before PPO's first iteration.
- `strategy` is currently `pretrain_then_rl`.
- `teacher.type` selects `astar`, `dijkstra`, `weighted_astar`, or
  `replanning_astar`.
- `teacher.weight` configures `weighted_astar` and must be null otherwise.
- `collection.episodes` is the required number of accepted demonstrations.
- `collection.seed_start` is the first deterministic environment reset seed.
- `collection.max_attempts` bounds failed or unsolved collection attempts.
- `collection.require_success` discards episodes that do not reach the goal.
- `collection.validation_fraction` reserves complete episodes for validation.
- `collection.reuse_dataset` reuses existing data only when its fingerprint
  matches every environment, task, observation, action, and teacher setting.
- `collection.dataset_dir` optionally points multiple runs or Tune trials to
  one shared compatible dataset. Generate it before concurrent trials start;
  incompatible environment or teacher settings are rejected by fingerprint.
- `pretraining.epochs` is the maximum number of behavior-cloning passes.
- `pretraining.batch_size` is the supervised optimizer batch size.
- `pretraining.learning_rate` applies only to behavior cloning.
- `pretraining.label_smoothing` configures categorical cross-entropy smoothing.
- `pretraining.early_stopping_patience` stops on stagnant validation loss.
- `handoff.initialize_encoder` retains learned observation features.
- `handoff.initialize_policy` retains learned policy/action-head parameters.
- `handoff.initialize_value_head` retains value parameters and defaults to
  false because action labels do not supervise values.

Demonstrations record the policy observation before each teacher action. The
dataset is split by episode, not by step. Artifacts are written under the run's
`imitation` directory: compressed data, manifest, epoch metrics, result, and
policy checkpoint.

Use `usage/experiments/train/ppo_tiny_overfit_imitation.yaml` for the first
validation and compare it with `ppo_tiny_overfit.yaml`.
