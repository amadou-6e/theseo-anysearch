# Waypoint curricula

Waypoint curricula train one start/goal stage at a time and may retain earlier stages in the training distribution. Stage advancement remains controlled independently by the training or evaluation `advance` block.

## Monotonic distance difficulty

Use `difficulty.mode: monotonic_distance` for an empty voxel grid:

```yaml
env:
  geometry:
    grid_size: 32
    boxes: []

  waypoint_curriculum:
    enabled: true
    initial_start: [16, 16, 16]
    initial_goal: [18, 18, 18]
    seed: 42

    difficulty:
      mode: monotonic_distance
      distance_increment: 2.0
      maximum_distance: null
      sampling_attempts: 512
```

The initial pair defines stage 0 and its Euclidean distance. Each later stage adds `distance_increment` voxel units.

While the requested distance is within the grid's inscribed radius, the start remains at the center. The goal direction is sampled uniformly on the sphere by normalizing three independent standard-normal values, then mapped to the nearest valid voxel displacement.

Beyond the grid radius, both endpoints can move throughout the grid. A spherical direction produces a displacement at the requested distance, and the start is sampled uniformly from positions for which the displaced goal remains in bounds. Distance continues increasing until `maximum_distance` or the grid diagonal. If `maximum_distance` is omitted, the diagonal is used.

`sampling_attempts` controls how many spherical candidates are considered before an exact lattice search chooses the closest valid displacement. Sampling is deterministic for a given curriculum seed and stage.

This strategy currently rejects STL, geometry-pool, and non-empty box geometry. Use the default `random` mode for environments containing static obstacles.

## Training-stage sampling

Training can sample any visited stage using normalized stage weights. The
legacy current/retained split remains supported when `strategy` is omitted.

```yaml
env:
  waypoint_curriculum:
    training_sampling:
      strategy: latest_multiplier
      latest_multiplier: 10.0
```

With four visited stages, this assigns raw weights `[1, 1, 1, 10]`, producing
probabilities of approximately `[0.077, 0.077, 0.077, 0.769]`.

The built-in strategies are:

```yaml
# Equal probability for every visited stage.
training_sampling:
  strategy: uniform

# Multiply only the latest stage's weight.
training_sampling:
  strategy: latest_multiplier
  latest_multiplier: 10.0

# Exponentially reduce weight with stage age.
training_sampling:
  strategy: recency
  recency_decay: 0.7
  minimum_weight: 0.1

# Emphasize stages with low cumulative evaluation success.
training_sampling:
  strategy: inverse_success
  minimum_weight: 0.1
  power: 1.0
  unevaluated_success_rate: 0.0
```

`inverse_success` uses cumulative deterministic retention-evaluation results.
An unevaluated stage uses `unevaluated_success_rate`. Raw weights are normalized
after every retention evaluation and broadcast to all rollout environments.
The probabilities used by each evaluation are recorded in its curriculum JSON.

A project may define a custom function in `curriculum_sampling.py` at its root:

```python
from theseo_anysearch.curriculum import StageSamplingContext, stage_sampling


@stage_sampling
def my_sampler(context: StageSamplingContext) -> dict[int, float]:
    return {
        stage.index: 10.0 if stage.is_latest else 1.0
        for stage in context.stages
    }
```

Select it by its Python function name:

```yaml
training_sampling:
  strategy: my_sampler
```

Custom functions return one finite, non-negative raw weight per stage. AnySearch
performs normalization and rejects missing, negative, non-finite, or all-zero
weights. `StageSamplingContext.stages` exposes the stage index, start and goal,
age, latest-stage flag, evaluation attempts, evaluation successes, and cumulative
evaluation success rate.

| Sampling parameter | Meaning |
|---|---|
| `strategy` | Built-in strategy or a registered function name. Defaults to `legacy`. |
| `latest_multiplier` | Raw weight assigned to the latest stage by `latest_multiplier`. |
| `recency_decay` | Per-stage age multiplier used by `recency`. |
| `minimum_weight` | Positive lower weight for `recency` and `inverse_success`. |
| `power` | Exponent controlling how strongly `inverse_success` prioritizes weak stages. |
| `unevaluated_success_rate` | Assumed success rate before a stage has evaluation results. |
| `current_stage_probability` | Legacy probability assigned to the current stage. |
| `retained_stage_probability` | Legacy aggregate probability divided equally among retained stages. |

## Parameters

| Parameter | Meaning |
|---|---|
| `mode` | `random` preserves native environment sampling; `monotonic_distance` enables the two-phase empty-grid strategy. |
| `distance_increment` | Euclidean voxel distance added after each stage transition. |
| `maximum_distance` | Optional distance cap; defaults to the grid diagonal and cannot be below the initial distance. |
| `sampling_attempts` | Number of uniformly directed candidates tested before the lattice fallback. |
