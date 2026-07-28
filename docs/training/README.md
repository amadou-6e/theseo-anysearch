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