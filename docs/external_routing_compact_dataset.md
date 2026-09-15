# External compact routing dataset preparation (#415)

## Develop Promotion Scope (#452)

This is retained historical encoder-study evidence, not a develop command
reference. The compact feature evaluator and preparation CLI remain on
`exp/perception-encoder`; they depend on its experimental encoder package.
Only the standalone routing contracts, exporters, route witnesses, provider
interface and shared collision checker are promoted here. See the original
[study implementation](https://github.com/amadou-6e/theseo-anysearch/tree/65a17a5/theseo_anysearch/garden)
for the commands below. Frozen study identities are unchanged.

Status: implementation and synthetic-fixture validation only. This is not a
training run, evaluation score, or claim that an external source is cleared for
training. Governing roadmap:
[`specs@1dc8397`](https://github.com/amadou-6e/specs/blob/1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b/projects/theseo-anysearch/routing-geometry-integration-roadmap.md).

`garden.external_routing` loads one-task #411 sidecar exports from #413-style
compact worlds. It verifies the source file bytes, immutable sidecar identities,
world `.npy` hash, task endpoints, route-artifact hash, and source rights for
`training`. It requires an independently validated reference record. A
feasible-only route or unreviewed source fails closed. This does not re-certify
source-geometry route replay; that claim remains with the source adapter.

The adapter builds `33^3` crops around each task endpoint. `raw_grid` has three
separate Boolean channels: observed occupied, observed free, and unknown.
Observation is a **synthetic** near-field plus 26-ray sensor, with rays ending at
the first occupied voxel. It is not a native simulator sensor or a partially
observed pathfinding episode. Full occupancy is used only to label straight
axis-aligned candidate segments of four or eight voxels. The swept-sphere/AABB
check treats the world boundary as solid; labels are occupancy-derived, not
independent source-CAD collision proofs. These labels define a narrow local
collision-readout task, not long-range path planning.

Plan JSON uses source and export paths relative to the plan file by default.
Use `--asset-root` to resolve them from a shared ignored asset directory when
running from a Git worktree; the committed #426 plan uses this option.

```json
{
  "governing_spec_sha": "1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b",
  "dataset_id": "external-compact-example-v1",
  "imports": [
    {
      "export": "exports/scene-01",
      "source_root": "sources/scene-source",
      "partition": "train"
    }
  ],
  "test_topology_families": []
}
```

```powershell
python -m theseo_anysearch.garden.external_routing_cli `
  --plan runtime/inputs/external-compact-plan.json `
  --output runtime/output/external-compact-v1
```

The output directory must not exist. `rows.npz` contains one observed input per
unique observation, query-to-observation indices, candidate offsets and binary
traversability labels; `rows.json` contains exact
world/task/observation/query/root/family/partition IDs. `prepare-report.json`
pins both artifacts by SHA-256 and records fresh dataset and query identities,
class counts and split coverage. Full-world truth stays in the original ignored
exports. `read_prepared_dataset` verifies the artifact hashes, arrays and query
identity before use. No raw grids, source assets or model weights enter Git.

Complete source layouts and sites must remain in one partition. Multiple
topology families require an explicitly named test-only family; a single family
can support only a narrower claim. A three-way train/calibration/test partition
is marked `three_way`; otherwise the report says `incomplete_not_fit_ready`.
Rows and all controls share the same query IDs, offsets and labels. For a
checkpoint supplied later, `extract_frozen_features` accepts only a fully
frozen compact encoder and returns its dense spatial volume and compact code
from observed channels. `bind_matched_controls` aligns these with raw grids and
shuffles codes only between **different root geometries in the same partition**.
It requires at least two roots per partition rather than silently using a
same-root or cross-partition donor. Neither function fits a head.

## Current readiness

The pinned #413 Aerial Gym source has a BSD-3-Clause top-level repository
license, but its adapter deliberately records selected asset rights as
`unreviewed`. A direct attempt to load the local derived altitude export for
training stops with `PermissionError: training is not cleared for this source`.
Asset-level rights and native sampled-scene parity remain [#423](https://github.com/amadou-6e/theseo-anysearch/issues/423).
The two #413 example layouts alone cannot populate three independent
partitions or two roots per partition. #414 is a separate single tunnel
generator family and does not yet carry an independently validated continuous
route reference for this adapter. Thus there is **no external-corpus training
manifest yet**. The no-network fixture suite validates the mechanism without
turning it into evidence for #416. Fresh source diversity, rights clearance,
the [specs#92](https://github.com/amadou-6e/specs/issues/92) preregistration,
and a separate compute authorization remain required before training.
