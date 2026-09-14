# Roofed Gazebo maze transfer tasks (#417)

This is a derived, collision-checked evaluation fixture governed by
[`amadou-6e/specs@1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b`](https://github.com/amadou-6e/specs/blob/1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b/projects/theseo-anysearch/routing-geometry-integration-roadmap.md).
The integration base was `exp/perception-encoder@4194a0f8f207c437674ba32fbcfb46fc7ebf97db`.
The original `easy_maze.world` has no roof and no route-query file. The
exporter preserves it as a separately identified open-top occupancy crop and
creates a roofed world and four fixed, derived queries. No source archive or
generated occupancy is committed.

## Source and rights

Source: [`engcang/gazebo_maps@6ecf68c`](https://github.com/engcang/gazebo_maps/tree/6ecf68cf659e122be58fc09c7a2af61b956c3301).
The pinned `3d_maze.tar.xz` is SHA-256
`5159e9f38b301307f71eccbe2ce101fd434398e60fd78ddd338e891fbb66deb1`;
`common_models.tar.xz` is
`03f46d349e46b55d4acba5c310d44205f1dab04ba8e107016dd4520551464a50`.
Both local files match the Git blobs at that commit. The repository's
[`LICENSE`](https://github.com/engcang/gazebo_maps/blob/6ecf68cf659e122be58fc09c7a2af61b956c3301/LICENSE)
is BSD-3-Clause. The source sidecar records a reviewed **evaluation-only**
use; this result does not authorize training or redistribute upstream files.

The converter resolves `easy_maze.world` includes through `model.config` and
the nested maze model. It uses 38 source box collisions including the
`grass_plane` ground, plus one `home` cylinder; `sun_2` is light only. It rejects
unresolved includes, unsupported collision shapes and pose frames, and archive
hash mismatches. Visual geometry is never used as collision truth.

## Fixed conversion

Run in an environment with the project dependencies installed:

```powershell
python -m theseo_anysearch.environments.gazebo_maze_export `
  --source runtime/research-assets `
  --output runtime/output/gazebo-g4-v1
```

The source directory must hold both pinned archives; output must not exist.
Storage is zero-based, `(x,y,z)`, right-handed and in source meters. Bounds
are `[-46,46] x [-46,46] x [0,9]` m at `0.5` m per voxel. The source world
remains open above; its finite crop is **not** a bounded flight task. The
separate roof has a collision underside at `z=8.0` m and thickness `1.0` m,
touching the source's 8 m main walls. A spherical body radius of `0.25` m is
used for query clearance and continuous swept-segment replay. Outside the
roofed world bounds is forbidden to the body.

The exact generated identities were:

| Record | SHA-256 |
| --- | --- |
| Converter source | `ff9bbfee68bf295ab345e5410320994ced91ea8bad754bc37015503fce67d86c` |
| Open-top world | `64637bb24dc5493faf4fecfc409b1182f9b29e18bedd15072c9e1115ab01d6dc` |
| Roofed world | `d5803eff9ea673b12716265278ca5c3b2d73195128f00f30348e8152abf4c180` |
| Open-top occupancy | `5c13595cf170df820c7b3fc7282c39819c055a77e748f0aca4dc73cea113ab60` |
| Roofed occupancy | `d4087a39f595d002af5704c2ac88d3798654e5ff8b1a3c1b1b797d72cc55b3c8` |
| Open-top world pack / file | `d0b0bb124b42fe434971e550876f126f7211ef9496fd39512a04862a02ac21e8` / `6541063fa04caf18d7e022689607a490b445ea7d80f3fd702de2ac6622a04be7` |
| Roofed world pack / file | `038e9795faa911e62024d35d1ab157d8da9f6ff90fbb5846c832f6c6c2467a16` / `1129fe4fc4d5744fd3b92d6d1f90b1b7ed9ce44278d84c526a564269a733550c` |
| Dataset bundle | `2489626b4b3d2583f8c389945a2f245e3fa43c88d15fa763df90a39f0e973452` |

The output directory contains the two `.npy` full-truth arrays, their compiled
world packs, source, conversion, world, task, reference and split sidecars,
route witnesses and a machine-readable `report.json`. It contains no
partial-observation input. Both
worlds have the same root geometry and test partition; the roof, body radius,
query endpoints and voxel scale participate in derived identities.

## Validation result

The fixed query list has four pairs. Three are accepted; `west_to_east` is
rejected because no six-axis body-valid route connects its endpoints. The
accepted `corner_to_corner` and `south_to_north` pairs have no same-altitude
route and require a 4 m altitude range. Their collision-replayed voxel-center
routes are 302.0 m and 275.5 m. `local_planar_control` has a same-altitude
route of 8.0 m. These are feasible witnesses, not certified continuous
optima or dynamic-flight trajectories.

The roofed grid has two six-connected free components and two body-valid
components. Of 431,200 free voxels, 332,130 pass the conservative radius
check; the largest body-valid component has 232,002 voxels. The minimum and
median reported body clearance are 0.274 m and 1.567 m. All 2,628 voxel
columns under the 24 main-wall collision shapes have zero body-valid
over-wall cells. Source-versus-voxel
checks found zero source-occupied/voxel-free samples in 10,000 random centers,
40 collision interiors, 427 surface probes, 8 endpoint openings, 6 border
probes and 132 diagonal probes. Boundary voxels are deliberately conservative;
1,054 random centers were voxel-occupied while source-free. Every accepted
route was replayed against original SDF primitives and the derived roof with
the spherical body radius. The reported 3D signal is limited to these two
fixed, derived altitude-change pairs in one maze layout, not cross-site
transfer or a Gazebo simulator score.

Validation: 31 offline tests passed across the Gazebo adapter and the shared
routing/Aerial Gym contracts. The ignored local export was re-run after the
final converter change; its report records member-level hashes and query IDs.
