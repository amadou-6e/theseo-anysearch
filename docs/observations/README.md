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
