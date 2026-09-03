# Composed sources

`env.geometry.sources` unions an STL room (`usage/geometries/corridor_l.stl`,
voxelized as an L-shaped wall the agent must route around) with one typed box
obstacle placed inside the open pocket. Neither source is aware of the other;
`resolve_geometry_sources` just takes the voxel union.

```powershell
anysearch geometry inspect usage/experiments/showcase/geometry_capabilities/01_composed_sources/experiment.yaml
anysearch geometry validate usage/experiments/showcase/geometry_capabilities/01_composed_sources/experiment.yaml --json
```

`validate` reports `geometry: valid`, `task: feasible`, and (in `--json`) a
39-step planned path that visibly detours around both the STL wall and the
box -- composition, not replacement.
