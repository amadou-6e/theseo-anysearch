# CaveDroneSim route witness and compact preparation (#426)

Develop promotion (#452) includes the standalone exporter and route checker.
The compact preparation command below belongs to the historical encoder study
on `exp/perception-encoder`, not to the develop CLI; its experimental encoder
dependencies are intentionally not promoted. Existing evidence is unchanged.

Status: CPU-only source export and dataset preparation, **not** model training,
an encoder result, a native flight task, or a cross-family transfer test.
Governing roadmap:
[`specs@1dc8397`](https://github.com/amadou-6e/specs/blob/1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b/projects/theseo-anysearch/routing-geometry-integration-roadmap.md).
This branch is temporarily stacked on `exp/415@bc0c447` while
[PR #425](https://github.com/amadou-6e/theseo-anysearch/pull/425) awaits review;
it does not authorize a merge or bypass [specs#92](https://github.com/amadou-6e/specs/issues/92).

The source, revision, rights and native bridge remain as recorded in
[the #414 audit](cave_drone_holdout.md). The pinned MIT source permits the
recorded evaluation and training uses. The source `World::Generate(seed)`
still defines the **complete voxel occupancy truth**. The #426 successor
finds a deterministic six-axis route through the conservative body-center
clearance mask for each fixed-goal pair, then separately checks every
center-to-center segment as a swept sphere against the source occupied voxel
cubes and solid world bounds. The checker computes exact segment-to-cube
distance in voxel units; it does not use the clearance mask or a learned
model. A failed replay aborts the export rather than silently dropping a
route. `reference-NN.json` and `route-*.json` are content-addressed; the task
movement model, task IDs and dataset IDs are fresh. World occupancy and its
identity are unchanged from #414. The independently validated reference means
**voxel-cube collision replay**, not continuous controller feasibility or
continuous-space optimality.

## Fixed local preparation

Six ignored exports were generated from pinned source commit
`ef7852198249390806d8c0cd42e576e02c73c19f` with 0.5 m voxels and a
0.25 m spherical body. Seeds 1–2 are train, 3–4 calibration, 5–6 test. Each
seed produced two tasks, two independently replayed references and no rejected
task stratum. Shortest six-axis route lengths and altitude range of the
altitude-target route were:

| Seed | Partition | Long-range steps | Altitude-target steps | Altitude range |
| --- | --- | ---: | ---: | ---: |
| 1 | train | 208 | 215 | 13.5 m |
| 2 | train | 209 | 204 | 12.0 m |
| 3 | calibration | 202 | 206 | 12.0 m |
| 4 | calibration | 208 | 202 | 12.5 m |
| 5 | test | 260 | 224 | 13.5 m |
| 6 | test | 213 | 203 | 12.0 m |

The committed [preparation plan](cavedrone-route-426-plan.json)
pins the seed-to-partition assignments and source paths relative to an
explicit `--asset-root`. From the workspace root, generate the six exports
under the named ignored paths, then prepare the corpus:

```powershell
$root = "C:\CodeWorkspace\theseo-anysearch"
foreach ($seed in 1..6) {
  $partition = if ($seed -le 2) {"train"} elseif ($seed -le 4) {"calibration"} else {"test"}
  python -m theseo_anysearch.environments.cave_drone_export `
    --source "$root\runtime\research-assets\cave-drone" `
    --output "$root\runtime\output\cavedrone-route-426-seed-$seed" `
    --seed $seed --partition $partition --body-radius-m 0.25
}
python -m theseo_anysearch.garden.external_routing_cli `
  --plan docs/cavedrone-route-426-plan.json `
  --asset-root $root `
  --output "$root\runtime\output\cavedrone-route-426-prepared"
```

The output directory must not already exist. The plan created **288** local
straight-segment query rows from **18** unique synthetic partial observations.
Every partition has two complete root geometries, so a same-partition,
different-root shuffled-code control is possible. The class counts are:

| Partition | Clear | Blocked |
| --- | ---: | ---: |
| Train | 42 | 54 |
| Calibration | 53 | 43 |
| Test | 50 | 46 |

The dataset identity is
`4b0eaefc2e94ceae292ea72876aa9e6a015da125726624ebd548b6e826d69e31`;
the query identity is
`6d43531f988d5d3641cc6b203a91f03aedf6d130d00d749ad6212482df35c1e6`.
`rows.json` hashes to
`ddc9eab62115b2c69f6753566db1b6d3aacea37af5d7338dfe12af68a77d69e9`;
`rows.npz` hashes to
`5f49c6357adae7bda5906cbc0629776775400b2b74f330b63e3d85341f6af4d6`.
A repeat preparation and the committed plan produced those same four hashes.
Generated worlds, routes, row arrays and the pinned source checkout remain
ignored local artifacts; no source or corpus is committed.

## Validity boundary

This is one upstream generator recipe across six seeds, not six topology
families. The only available comparison would be **within-family local
collision prediction** under a synthetic 26-ray-plus-near-field visibility
model. It is not a flight observation model, long-range pathfinding result,
cross-family generalization result, or evidence that a pretrained encoder is
better than raw voxels. Six roots and twelve fixed tasks are a plumbing slice,
not a statistical study. Before #416 can fit a model, review and merge the
stack, expand and audit layout diversity, freeze a new governing
preregistration under specs#92 with data/query IDs, model/controls, geometry
uncertainty and compute caps, and obtain an explicit compute authorization.
