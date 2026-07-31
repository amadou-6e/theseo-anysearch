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
| `custom` | `null` | Name of a function in `rewards.py` or `rewards.rs`. |

Progress shaping is:

```text
distance_shaping × (previous distance − new distance)
```

A move toward the goal therefore receives a less negative or more positive result than a move away. The constant `step_cost` is still applied independently. Zone rewards must remain negative and satisfy `zone_reward_min <= zone_reward_max < 0`.

Legacy reward fields such as `env.step_cost` and `env.goal_reward` remain loadable during migration. Do not combine them with `env.rewards` in the same configuration.

## Named custom rewards

Select a reward by name in YAML:

```yaml
env:
  rewards:
    custom: collision_aware
```

The same selector is used for Python and Rust definitions. A Python experiment
places the named function in `rewards.py` beside its YAML:

```python
from theseo_anysearch.experiments.custom_rewards import RewardResult


def collision_aware(context):
    penalty = -0.02 if context.collision else 0.0
    return RewardResult(
        reward=penalty,
        components={"extra_collision_penalty": penalty},
        mode="add",
    )
```

A compiled extension defines the same name in `extension/src/rewards.rs`:

```rust
use anysearch_extension::{anysearch_reward, RewardContext, RewardResult};

#[anysearch_reward]
pub fn collision_aware(context: &RewardContext) -> RewardResult {
    let penalty = if context.collision { -0.02 } else { 0.0 };
    RewardResult::add(penalty).with_component("extra_collision_penalty", penalty)
}
```

The attribute generates `anysearch_reward_collision_aware_v1` automatically.
There is no handwritten ABI wrapper in `lib.rs`. When both Python and Rust
implement the selected name, the compiled Rust definition takes precedence. It
is loaded once by the Rust core and called directly inside `VoxelEnv::step`; the
per-step reward does not pass through Python. Python remains the fallback when
the compiled extension does not supply a reward capability.

`mode="add"` retains built-in components and appends custom components.
`mode="replace"` discards built-in components. Rust owns and validates the final
named breakdown, and Python exposes it unchanged through Gymnasium `info`.

The selected `rewards.py` or compiled library is archived in every ordinary run
or Tune trial. See [`rewards.py`](../../usage/experiments/showcase/rewards.py) and
the [`native_extension`](../../usage/experiments/showcase/native_extension)
example.
