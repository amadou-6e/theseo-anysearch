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

## Parameters

| Parameter | Meaning |
|---|---|
| `mode` | `random` preserves native environment sampling; `monotonic_distance` enables the two-phase empty-grid strategy. |
| `distance_increment` | Euclidean voxel distance added after each stage transition. |
| `maximum_distance` | Optional distance cap; defaults to the grid diagonal and cannot be below the initial distance. |
| `sampling_attempts` | Number of uniformly directed candidates tested before the lattice fallback. |
