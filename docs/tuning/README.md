# Tuning configuration

A `tune_config` block turns an experiment into a Ray Tune hyperparameter search while retaining the environment, training, algorithm, and model defaults from the surrounding YAML.

```yaml
tune_config:
  scheduler: asha
  num_samples: 4
  metric: evaluation_success_rate
  mode: max
  max_concurrent: 2
  checkpoint_frequency: 1
  preserve_trial_artifacts: true
  search_space:
    algorithm_config.lr:
      type: loguniform
      lower: 0.0001
      upper: 0.001
```

## Fields

| Field | Default | Purpose |
|---|---:|---|
| `scheduler` | `asha` | Selects `asha`, `pbt`, `random`, `hyperband`, `bohb`, `optuna`, `cmaes`, or `flaml`. |
| `num_samples` | `10` | Number of sampled trials. |
| `metric` | `episode_reward_mean` | Scalar metric optimized by Tune. |
| `mode` | `max` | Selects maximization or minimization. |
| `max_concurrent` | `4` | Maximum simultaneously running trials. |
| `search_space` | `{}` | YAML-friendly parameter distributions and choices. |
| `checkpoint_frequency` | `1` | Tune report interval for restorable checkpoints. |
| `max_environment_steps` | `null` | Optional sampled-environment-step budget per trial. |
| `max_wall_time_s` | `null` | Optional wall-clock budget per trial. |
| `target_success_rate` | `null` | Stops a trial after deterministic evaluation reaches this rate. |
| `preserve_trial_artifacts` | `true` | Retains checkpoints and replay artifacts for stopped or pruned trials. |
| `ray_storage_dir` | system temp | Optional short Ray Tune storage path. |
| `ray_temp_dir` | system temp | Optional short Ray runtime path. |

## Scheduler settings

Scheduler-specific options belong in the matching subblock:

```yaml
tune_config:
  scheduler: asha
  asha_config:
    max_t: 40
    grace_period: 10
    reduction_factor: 3
    brackets: 1
```

Available subblocks are `asha_config`, `pbt_config`, `hyperband_config`, `bohb_config`, `optuna_config`, `cmaes_config`, and `flaml_config`. See [Tune lifecycle and trial budgets](lifecycle.md) for reporting, pruning, checkpoint, artifact, and resume behavior.