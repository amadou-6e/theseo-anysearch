# Sample Geometries

ASCII STL files for use with `anysearch experiment run --config <yaml>`.
All units are arbitrary (the voxelizer is scale-invariant via `env.scale`).

| File | Triangles | Description | Recommended use |
|---|---|---|---|
| `cube.stl` | 12 | Single 50×50×50 box | Smoke test; minimal surface area |
| `stepped_terrain.stl` | 300 | 5×5 grid of boxes at varying heights (10–30 units) | Surface pathfinding baseline; tests height transitions |
| `corridor_l.stl` | 24 | Two rectangular corridors forming an L-shape | Routing test; single clear path, forced turn |
| `pipe_junction.stl` | 60 | Central hub with four radiating branches | Multi-route routing; agents must choose branch |
| `ramp_spiral.stl` | 96 | Single-turn helical ramp rising 60 units | 3-D surface pathfinding; tests vertical routing |

## Usage

```yaml
env:
  stl_path: usage/geometries/stepped_terrain.stl
  scale: 40.0        # voxelization resolution — higher = finer grid, slower
  agent_count: 4
  max_steps: 200
```

`scale` controls how many voxels per unit length. A 50-unit cube at `scale: 40` produces
a 2000-unit voxel bounding box. Start with `scale: 20–40` for fast iteration.
