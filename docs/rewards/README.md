# Reward configuration

All reward terms belong under `env.rewards`.

```yaml
env:
  rewards:
    step_cost: -0.01
    collision_cost: 0.0
    goal_reward: 1.0
    distance_shaping: 0.2
    distance_reward_mode: progress
```

## Fields

| Field | Default | Purpose |
|---|---:|---|
| `step_cost` | `-0.01` | Constant reward applied on every step. |
| `collision_cost` | `0.0` | Additional term for a blocked move. |
| `goal_reward` | `1.0` | Terminal reward added when the goal is reached. |
| `distance_shaping` | `0.0` | Weight applied to distance progress. |
| `distance_reward_mode` | `progress` | Selects progress shaping or always-negative distance zones. |
| `zone_reward_min` | `-1.0` | Most negative reward in zone mode. |
| `zone_reward_max` | `-0.01` | Least negative reward in zone mode. |
| `zone_reward_curve` | `linear` | Selects linear or exponential zone interpolation. |
| `distance_metric` | `euclidean` | Selects Euclidean or Manhattan goal distance. |
| `invalid_action_cost` | `0.0` | Additional term for an invalid action. |
| `construction_residual_weight` | `0.0` | Weight for remaining construction work. |
| `construction_overshoot_weight` | `0.0` | Weight for construction beyond the requested target. |

Progress shaping is:

```text
distance_shaping × (previous distance − new distance)
```

A move toward the goal therefore receives a less negative or more positive result than a move away. The constant `step_cost` is still applied independently. Zone rewards must remain negative and satisfy `zone_reward_min <= zone_reward_max < 0`.

Legacy reward fields such as `env.step_cost` and `env.goal_reward` remain loadable during migration. Do not combine them with `env.rewards` in the same configuration.

## Custom Python rewards

An experiment can replace or extend the built-in YAML reward without configuring an import path. For `example.yaml`, Anysearch looks beside the YAML for `reward.example.py`, then falls back to `reward.py`.

The module defines:

```python
from theseo_anysearch.experiments.custom_rewards import RewardResult


def compute_reward(context):
    penalty = -0.02 if context.collision else 0.0
    return RewardResult(
        reward=penalty,
        components={"extra_collision_penalty": penalty},
        mode="add",
    )
```

`RewardContext` exposes the action, previous/current observations and cursor positions, goal and distance values, termination flags, standard reward and breakdown, task information, and runtime environment configuration.

`mode="add"` adds the returned reward and components to the standard reward. `mode="replace"` discards the standard reward and uses only the returned value and components. Component values must be finite, use Python identifier names, sum to `reward`, and not collide with built-in component names.

The selected source is archived as `reward.py` in every ordinary run or Tune trial. `custom_reward.json` records its SHA-256 hash. Single-agent Ray training and evaluation workers receive the archived absolute path, so they execute the same source captured by the run.

See [`reward.quick_demo.py`](../../usage/experiments/showcase/reward.quick_demo.py) for a complete additive example.
