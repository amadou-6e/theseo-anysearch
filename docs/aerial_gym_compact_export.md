# Compact Aerial Gym collision export (#413)

This optional adapter exports two **derived static tasks**, not episodes from
the Isaac Gym simulator. It uses the collision boxes in seven pinned upstream
URDFs, the six source wall placement ratios, and a deterministic local recipe
for obstacle instances. `detour` blocks the direct route; `altitude` makes a
full-width low barrier. Both include two seed-controlled side obstacles.
No visual mesh or robot model is used. Arbitrary URDF joints, meshes, and
non-box collision primitives fail closed.

The source is [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator)
at `f0d0f05283f7897bab5a1bcc7b19b91cebbab218`; its
[navigation configuration](https://github.com/ntnu-arl/aerial_gym_simulator/blob/f0d0f05283f7897bab5a1bcc7b19b91cebbab218/aerial_gym/config/env_config/env_with_lidar_nav_obstacles.py)
sets a bounded room and includes panels, objects and six walls. The upstream
[asset configuration](https://github.com/ntnu-arl/aerial_gym_simulator/blob/f0d0f05283f7897bab5a1bcc7b19b91cebbab218/aerial_gym/config/asset_config/lidar_nav_env_config.py)
and [asset manager](https://github.com/ntnu-arl/aerial_gym_simulator/blob/f0d0f05283f7897bab5a1bcc7b19b91cebbab218/aerial_gym/env_manager/asset_manager.py)
sample actor states at runtime. The export does **not** replay those samples;
its fixed starts, goals, and obstacles have new identities and a `derived`
provenance. Source config and every selected URDF have SHA-256 entries in the
generated `source.json`; `scene-instances.json` records every resolved pose.
If the source root is a Git checkout, the exporter verifies its HEAD and
selected-file cleanliness against `--source-revision`.

## Export

Clone the source outside Git-tracked project paths, then run:

```powershell
python -m theseo_anysearch.environments.aerial_gym_export `
  --source-root runtime/research-assets/aerial_gym_simulator `
  --source-revision f0d0f05283f7897bab5a1bcc7b19b91cebbab218 `
  --output runtime/output/aerial-gym-altitude-seed3 `
  --seed 3 --layout altitude
```

The output directory must not exist. Its `occupancy.npy` is full-world truth
in storage-axis `(x,y,z)` order, 0 free/1 occupied, at 0.25 m per voxel by
default. It is not a sensor observation. The `source`, `conversion`, `world`,
`task`, `reference`, and `split` sidecars use the #411 version-1 contracts. The conversion
identity includes the pose-file hash, rasterization version, seed, layout,
voxel size and body radius. `export-report.json` records route checks but is
not a certified-optimal route reference. `route-storage.json` holds a witness
whose six-axis segments are replayed against the original transformed URDF
collision boxes expanded by body radius. Its `independently_validated` claim
means collision-checked, not globally optimal in continuous space.

Each collision box is transformed by the actor and URDF collision origins.
Voxelization marks cells that may intersect a box (conservative local-axis
bound). For the 6-axis route check, each box is expanded by the body radius
before voxelization. This yields a conservative continuous box-clearance
witness for grid-center moves; it is **not** a validated dynamic flight trajectory,
dynamic-obstacle episode or performance comparison with the upstream task.

## Verified local slice

At the pinned source commit, seed 3, 0.25 m voxels and 0.25 m body radius:

| Layout | Extent | Collision boxes | Occupied cells | Straight cells | Shortest 6-axis cells | Same-altitude path | World ID prefix |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `detour` | 40 x 40 x 24 | 9 | 6,812 | 16 | 24 | yes | `351597da` |
| `altitude` | 40 x 40 x 24 | 48 | 7,956 | 24 | 48 | no | `a9b9c9d5` |

A repeat export of the altitude case matched all generated file hashes.
Synthetic unit fixtures check collision origins, rotations, source/voxel
agreement at voxel centers, unsupported primitives, immutable sidecars and
layout topology without downloading upstream files in CI. The source root,
converted grids and poses are ignored runtime artifacts, not committed assets.

The upstream repository's top-level `LICENSE` states BSD 3-Clause, but this
adapter deliberately writes `rights.status=unreviewed` and no allowed uses
until asset-level rights and intended training/redistribution scope are
reviewed. The presence of exported worlds does **not** authorize training.
Before #415/#416, review that gate, inspect generated layout diversity and
freeze fresh external-corpus identities under a merged governing spec.
