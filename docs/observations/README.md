# Observation configuration

Policy observation settings belong under `env.observation`.

```yaml
env:
  observation:
    mode: box
    box_radius: 1
```

## Fields

| Field | Default | Purpose |
|---|---:|---|
| `mode` | `scalar` | Selects `scalar`, `box`, `radial`, or `hierarchical_box` observations. |
| `box_radius` | `2` | Radius of a local cubic observation. Radius 1 produces a `3 × 3 × 3` grid. |
| `box_radii` | `null` | Radii concatenated by hierarchical box observations. |
| `ray_max_len` | `16` | Maximum radial ray distance. |

Examples:

```yaml
# Radius-1 local occupancy box
env:
  observation:
    mode: box
    box_radius: 1

# Radial observation
env:
  observation:
    mode: radial
    ray_max_len: 10

# Multiple local scales
env:
  observation:
    mode: hierarchical_box
    box_radii: [1, 2, 4]
```

Legacy fields such as `env.obs_mode`, `env.box_radius`, and `env.ray_max_len` remain loadable during migration. Do not combine them with `env.observation` in the same configuration.

## Shared navigation features

Every observation mode includes `goal_direction`, the Euclidean unit vector
from the cursor to the active goal. It is `(0, 0, 0)` at the goal. Goal
distance is represented separately by `goal_distance`, so changing only the
distance does not change `goal_direction`.

This unit-vector encoding replaces the earlier grid-scaled displacement.
Existing policy checkpoints trained with the earlier encoding are therefore
not observation-compatible and should be retrained.

## No absolute-position or time-budget inputs

Policy observations do not include `cursor_pos` or `steps_remaining`. The
environment continues to track both values internally for movement, episode
termination, rewards, predicates, rendering, and trajectory recording, but the
network cannot use them as shortcuts.

Removing these fields is an intentional schema break. Checkpoints whose
observation space contains either field are incompatible and must be retrained;
AnySearch does not silently pad or discard checkpoint inputs.

Without a configured goal, `scalar` mode consequently has no policy features.
Use a spatial observation mode or a goal-conditioned task for trainable runs.
