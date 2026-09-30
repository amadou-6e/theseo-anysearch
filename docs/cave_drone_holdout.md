# CaveDroneSim tunnel-world export (#414)

Status: source adapter and audit, based on integrated #411 at
`0f6bc476de2fe9ab3d7678a3c1eb52c9cb2ac154`. This is not a frozen
training corpus or an independent topology-family holdout. Governing roadmap:
`amadou-6e/specs@1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b`.
Issue #414 records the earlier review-stage stacking deviation.
This document describes the original #414 integration; the
[#426 successor](cavedrone_route_witness.md) adds route witnesses and a
six-seed preparation slice without changing the original evidence.

## Source and rights

- Source: [makarov-mm/cave-drone](https://github.com/makarov-mm/cave-drone),
  commit `ef7852198249390806d8c0cd42e576e02c73c19f`.
- Source `LICENSE` is MIT, copyright 2026 Mykhailo Makarov. The exporter
  verifies the revision, clean tracked source, exact notice and source-file
  hashes. It records evaluation and training as reviewed uses, but does not
  commit or redistribute the source or generated worlds.
- `World::Generate(seed)` is called from a small C++ bridge compiled against
  the pinned upstream `World.cpp`; no replacement geometry algorithm is used.
  Upstream dimensions are 192 x 56 x 192 in x/y/z storage order, with y
  vertical and 0.5 m voxels. The exported `uint8` NPY is **complete occupancy
  truth**, not a simulated LiDAR observation.

## Reproduce

After cloning the pinned upstream source outside Git's tracked tree, run from
this repository with a C++23 compiler available:

```powershell
python -m theseo_anysearch.environments.cave_drone_export `
  --source C:\path\to\cave-drone `
  --output C:\path\to\ignored-output\seed-1 `
  --seed 1 --partition test --body-radius-m 0.25
```

The command writes `occupancy.npy`, content-addressed source/conversion/world/
task/split sidecars and `report.json`; it refuses to overwrite an output
directory. Every source seed is a distinct root geometry. Derived task starts
and goals are voxel centers; they are not CaveDroneSim's native continuous
exploration start/return mission. The fixed-goal selection requires at least
8 m endpoint separation, a six-connected clearance component, and attempts a
separate goal with at least 2 m y displacement. Missing strata are recorded
as rejections. Clearance conservatively bounds a spherical body against the
occupied voxel cubes, map boundary and every six-neighbor centerline edge.
The original #414 export claimed no route witness, continuous controller
validation or sensor observation. The #426 successor adds a voxel-cube
collision-replayed route witness, not controller or sensor validation.

## Local source census

The following ignored outputs were generated from that pinned source with
body radius 0.25 m. Re-exporting seed 1 twice produced identical NPY bytes,
occupancy hash and sidecar identities in the local integration test.

| Seed | Occupancy SHA-256 | Free cells | Six-connected free components | Clearance-passable cells | Clearance components | Start component cells |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `1a44460b10a814ae1bd8f1fca5c4df30b5693536f32b5b62a5aa3e63b5f6e99a` | 586,757 | 1 | 289,118 | 258 | 286,167 |
| 2 | `3db3c84ac2a4c4a2294522f5997d38d2f9d2d1197a9d2606052e543d38fc53aa` | 552,272 | 1 | 262,079 | 262 | 253,572 |
| 3 | `b85e97c202d641bf5c27a8e793a1eb6728842e0acd7288ab0bc09fb4b9db7d8c` | 583,787 | 1 | 286,129 | 296 | 279,290 |

The native generator flood-fills away disconnected voids, so one point-free
component per seed is expected. Finite-radius erosion creates many small
clearance components, while most safe cells remain in the start component.
The counts are a smoke-test census, not evidence of task difficulty or
cross-family generalization.

## Holdout decision

The pinned upstream code has one fixed generator recipe: fBm chambers,
14 random-walker tunnels and one post-generation cavity filter. Changing its
seed changes a layout **within that same family**. The output sidecars name
that family explicitly and set `cross_family_holdout_supported: false`.
Train/validation/test seed assignments can test within-family variation only.
A claim about held-out chamber-versus-tunnel topology requires another source
family or a separately versioned, reviewed generator intervention and fresh
dataset identities. #414 should therefore be retained as a faithful adapter,
not accepted as completing the cross-family holdout gate.
