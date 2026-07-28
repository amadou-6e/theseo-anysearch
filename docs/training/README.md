# Training configuration

The top-level `training` block controls execution of one resolved experiment. Algorithm hyperparameters remain in `algorithm_config`, and neural-network settings remain in `model_config`.

```yaml
training:
  algorithm: ppo
  model: fcnet
  runner: local
  iterations: 80
  checkpoint_interval: 20
  require_gpu: true
  num_env_runners: 8
  trajectory_every: 10
  best_trajectory: true
  video_every: 0
```

## Fields

| Field | Default | Purpose |
|---|---:|---|
| `algorithm` | required | Trainer family, such as `ppo`, `dqn`, `sac`, `rainbow`, `multi_agent_voxel_ppo`, or `heuristic`. |
| `model` | `voxel_encoder` | Model configuration family loaded from `model_config`. |
| `runner` | `local` | Selects local or Anyscale execution. |
| `iterations` | `100` | Maximum number of completed training iterations. |
| `checkpoint_interval` | `10` | Iteration interval for policy checkpoints. |
| `require_gpu` | `false` | Fails startup when a GPU is required but unavailable. |
| `num_gpus` | automatic | Explicit RLlib GPU allocation override. |
| `num_env_runners` | `0` | Parallel rollout workers; zero samples in the local trainer process. |
| `trajectory_every` | `10` | Iteration interval for replayable evaluation trajectories. |
| `best_trajectory` | `true` | Retains the best evaluation trajectory observed so far. |
| `video_every` | `10` | Iteration interval for rendered video artifacts. |
| `evaluation_episodes` | `1` | Deterministic evaluation episodes collected per iteration. |
| `evaluation_seed` | `42` | First deterministic evaluation seed. |
| `evaluation_min_success_rate` | `0.5` | Success-rate threshold used to classify an evaluation as solved. |
| `early_stop` | disabled | Optional deterministic-evaluation condition for ending training before `iterations`. |

## Related blocks

```yaml
algorithm_config:
  lr: 0.0003
  gamma: 0.99
  train_batch_size: 1024

model_config:
  hidden_sizes: [256, 256]
  activation: relu
```

Algorithm-specific models validate additional fields such as PPO clipping, minibatch, and SGD settings. See [goal-finding evaluation](evaluation.md) for deterministic evaluation behavior and reported metrics.
## Early stopping

Standard training can stop before its hard `iterations` limit when deterministic
evaluation repeatedly reaches a target:

```yaml
training:
  iterations: 100
  evaluation_episodes: 20

  early_stop:
    enabled: true
    mode: goal_finishes
    min_iterations: 5
    min_consecutive_evaluation: 3
    min_goal_finishes: 20
```

The available modes and their matching thresholds are:

| Mode | Required field | Condition |
|---|---|---|
| `reward` | `min_reward` | Mean deterministic evaluation reward is at least the threshold |
| `goal_finishes` | `min_goal_finishes` | Goals completed in the current evaluation batch are at least the threshold |
| `heuristic_accuracy` | `min_heuristic_accuracy` | Exact next-action agreement with the same-seed heuristic is at least the threshold |
| `heuristic_distance` | `max_heuristic_distance` | Mean distance between policy and heuristic next-block offsets is at most the threshold |

Exactly one threshold is used for the selected mode. `min_iterations` delays
condition counting, while `min_consecutive_evaluation` requires the condition to
remain true across multiple evaluation batches. A failed check resets the
consecutive count. `iterations` remains the hard maximum.

Heuristic comparison can select `astar`, `dijkstra`, `weighted_astar`, or
`replanning_astar`:

```yaml
training:
  early_stop:
    enabled: true
    mode: heuristic_accuracy
    min_heuristic_accuracy: 0.95
    heuristic_type: weighted_astar
    heuristic_weight: 1.5
    min_iterations: 10
    min_consecutive_evaluation: 3
```

Action comparison uses canonical 26-neighbor indices, so `vector_3` and discrete
policies are compared consistently. `heuristic_accuracy` is exact agreement.
`heuristic_distance` distinguishes near misses from opposite moves:

```yaml
training:
  early_stop:
    enabled: true
    mode: heuristic_distance
    heuristic_distance_metric: l1  # or l2
    max_heuristic_distance: 0.25
    min_iterations: 10
    min_consecutive_evaluation: 3
```

The distance is measured between the two next-move offsets in voxel units. For
example, `(1, 0, 0)` versus `(1, 1, 0)` has both L1 and L2 distance `1`;
`(1, 0, 0)` versus `(-1, 1, 0)` has L1 distance `3` and L2 distance `sqrt(5)`.
The reported value is the mean across compared steps, and lower is better.
Heuristic comparison currently supports the single-agent environment.

An early-stopped run remains `COMPLETED`. It writes `early_stop.json`, preserves
its consecutive state in `early_stop_state.json`, forces a final trajectory, and
saves a checkpoint at the stopping iteration. The final `run.json` records the
reason, iteration, achieved value, and threshold.