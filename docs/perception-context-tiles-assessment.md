# Frozen tiled spatial coverage assessment

Issue #376. Disposition retain as a memory option, not a speed optimization.
Source c4f1787; authorized unmerged stack on #374. No merges or promotion.
Spec amadou-6e/specs@76bb3c403fb36b30ebf5335dcbc8f77005cf2db4,
`projects/theseo-anysearch/python/perception-context-tiles.md`.
Run voxel-context-tiles-v1-run1; no encoder or probe fitting.

## Quality

All four checks pass against both dense controls on fresh48 parents, querying
central49 within65. Existing weights, normalization and thresholds unchanged.
27 tiles of33, four per batch, core ownership16/16/17 along each axis, halo8.

| Task | Tiled33/head33 seed scores | Gain CI vs dense65/head33 | Gain CI vs dense65/head65 |
| --- | --- | --- | --- |
| Occupied IoU | .6509/.6776/.6760 | [-.00284,.00664] | [.00022,.00931] |
| Boundary F1 | .6905/.6838/.7021 | [-.00444,.00890] | [.00182,.01702] |
| Clearance NMAE | .03826/.04201/.04241 | [.00186,.00258] | [.00213,.00278] |
| Recovery NMAE | .03615/.04138/.04188 | [.00196,.00263] | [.00213,.00281] |

Improvement reverses error signs. Paired parent bootstrap, six strata,2000 draws,
three fixed encoder seeds, margins .03/.02. Same-head comparison isolates the
inference-layout change: surface differences inconclusive, distance modestly
better tiled. Against the adapted65 head, all intervals are positive, but these
multiple pilot diagnostics are not multiplicity-adjusted certification. No
parameter/threshold tuning occurred on assessment. Family scores and all dense
controls are in the machine report. Quality is sampled, not exhaustive49 output.

## Resources and decision

| Encoder coverage path | ms/scene range | Total allocated MiB | Incremental MiB |
| --- | --- | --- | --- |
| Dense65, batch1 scene | 21.8-23.5 | 176.60 | 142.36 |
| Tiled33, batch4 tiles | 96.9-104.7 | 112.19 | 74.64 |

Tiling costs about4.1-4.8 times latency for36.5% lower total allocation. These
profiles cover every tile, excluding input transfer, heads and stitching. Local
timings, not isolated production performance or total GPU device memory. Previous
#374 profiled batch4 scenes; do not compare its absolute peaks as batch1 results.

**Direction:** retain the shared frozen encoder and33-context heads; use dense
evaluation when memory fits. Retain tiling only as a bounded-memory alternative,
not the default speed path. More dense context did not help central queries in
#374; overlap tiling does not make wider coverage cheaper here. Do not launch a
larger dense model solely because65 fits. A future scaling experiment must test
a specific resource/coverage need (such as fewer overlapping tiles or sparse
multiscale context), with new identities, not continue increasing volume blindly.
This resolves the immediate context/tiling choice without claiming radius512,
topology, pathfinding or universal encoder readiness.

## Evidence

Elapsed51.812 seconds including registration preparation. Full garden418 passed
before the added completed-report replay test. Exhaustive query-to-tile mapping
test covers49 cubed cells. Source24 probe artifacts and encoder state hashes
verified; no optimization calls in this run.36 prediction artifacts remain outside
Git; compact report records their hashes. Fresh parent hashes are disjoint from
all context65 training/selection/assessment parents.

Registration: `perception-encoder-local-geometry/tiles-preregistration.json`.
Report: `perception-encoder-local-geometry/tiles-report.json`.
Payload SHA256 `0923d9216166f5e2e9ef259a68b7f1723f1d848e4b5486e9d6830dcb8e740ed2`.
