# Perception configuration

Policy observations are ordered compositions of named sensors under
`env.observation.sensors`. Each sensor declares its output space, dtype,
normalization, required environment capabilities, compatible models, estimated
cost, and schema version before training begins.

```yaml
env:
  observation:
    sensors:
      - name: local_box
        parameters:
          radius: 1
      - name: radial
        parameters:
          max_len: 10
      - name: goal_vector
```

The resulting Gymnasium observation is a structured dictionary with stable
feature keys. Sensor order is deterministic. Unknown sensors, duplicate sensors,
duplicate feature keys, invalid parameters, missing capabilities, and known
model incompatibilities fail while settings are loaded, before Ray starts.

## Built-in sensors

| Sensor | Features | Parameters | Normalization |
|---|---|---|---|
| `goal_vector` | `goal_distance` (1), `goal_direction` (3) | none | distance divided by maximum grid Manhattan distance; direction in `[-1, 1]` |
| `local_box` | `local_grid` | `radius` (default 2) | binary occupancy in `[0, 1]`, flattened in x/y/z order |
| `radial` | `ray_hits`, `ray_hit_types` (26 each) | `max_len` (default 16) | proximity and hit type in `[0, 1]` |
| `hierarchical_box` | `local_grid` | `radii` (default `[1, 4]`) | concatenated binary boxes in configured radius order |

`goal_vector` requires configured goal-capable geometry. Local and hierarchical
box sensors declare compatible spatial model families, so an incompatible custom
encoder is rejected during preflight validation.

## Legacy migration

`observation.mode` and flattened `env.obs_mode` remain compatibility aliases and
emit `DeprecationWarning`. They resolve as follows:

| Legacy mode | Equivalent sensors |
|---|---|
| `scalar` | `goal_vector` |
| `box` | `local_box`, `goal_vector` |
| `radial` | `radial`, `goal_vector` |
| `hierarchical_box` | `hierarchical_box`, `goal_vector` |

Legacy radius fields are translated into sensor parameters. Do not combine
`sensors` with `mode`, or mix flattened observation fields with the nested
`observation` block.

## Third-party sensor

Third-party code subclasses `Perception`, declares metadata and space/compute
operations, then registers once before loading settings:

```python
import numpy as np
from gymnasium import spaces

from theseo_anysearch.environments.perception import (
    Perception,
    PerceptionMetadata,
    PerceptionParameters,
    register_perception,
)


class FilledRatioParameters(PerceptionParameters):
    include_agent_cells: bool = True


class FilledRatio(Perception):
    metadata = PerceptionMetadata(
        name="filled_ratio",
        normalization="fraction in [0, 1]",
        required_capabilities=("voxel_grid",),
    )
    parameter_model = FilledRatioParameters

    def observation_space(self, env):
        return {"filled_ratio": spaces.Box(0.0, 1.0, (1,), np.float32)}

    def compute(self, env, native_observation):
        filled = len(env._rust_env.filled_voxels())
        total = env._config["grid_size"] ** 3
        return {"filled_ratio": np.asarray([filled / total], dtype=np.float32)}


register_perception("filled_ratio", FilledRatio)
```

The environment and trainers require no edits. Extension packages should test
that emitted values satisfy the declared Gymnasium space for reset and step.

## Shared navigation features

`goal_direction` is the Euclidean unit vector from cursor to goal and becomes
`(0, 0, 0)` at the goal. `goal_distance` is separate, so a distance change does
not alter direction. Without a configured goal, explicitly selecting
`goal_vector` is invalid.

Absolute cursor position and remaining time are intentionally not exposed to the
policy. Existing checkpoints containing those removed fields are incompatible
and must be retrained.
