# Geometry configuration

Geometry sources and voxelization settings belong under `env.geometry`.

```yaml
env:
  geometry:
    stl_path: usage/geometries/stepped_terrain.stl
    scale: 40.0
    grid_size: 32
    padding: 2
```

## Fields

| Field | Default | Purpose |
|---|---:|---|
| `stl_path` | `null` | One STL file voxelized for the environment. |
| `stl_paths` | `null` | Multiple STL files used to construct a geometry pool. |
| `scale` | `1.0` | Voxelization scale for a fixed STL. |
| `scale_range` | `null` | Minimum and maximum scale used for geometry variation. |
| `grid_size` | `32` | Side length of the cubic voxel grid. |
| `boxes` | `null` | Procedural boxes in `[xmin, ymin, zmin, xmax, ymax, zmax]` form. |
| `pool_size` | `0` | Number of procedural geometries prepared for sampling. |
| `scale_variants_per_map` | `4` | Number of scale variants generated per STL map. |
| `padding` | `2` | Free-space padding around imported geometry. |
| `pool` | `null` | Precomputed geometry-pool and augmentation settings. |

A precomputed pool can be configured as follows:

```yaml
env:
  geometry:
    grid_size: 64
    pool:
      pool_dir: runtime/geometry_pools/highres
      augmentation:
        paste_boxes:
          num_boxes: [2, 12]
          box_min_size: [2, 2, 2]
          box_max_size: [20, 20, 20]
          prob: 1.0
```

Legacy fields such as `env.stl_path`, `env.grid_size`, and `env.geometry_pool` remain loadable during migration. Do not combine them with `env.geometry` in the same configuration.