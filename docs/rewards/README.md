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