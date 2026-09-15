# V2 foundation result

Run `voxel-encoder-pilot-v2-p0-calibration-1` completed under specification
commit `01eefc529016da48c4a1dd17b85391720542af14` and returned `blocked`.
The content-addressed report payload is
`a7a149f9235b38f9ff1f1a230ce791367cf528fc6705e0f987674f1a48d4ea43`.

The measured denominator failures were:

| Component | Best floor | Ceiling | Result |
| --- | ---: | ---: | --- |
| `occupied_iou` | 1.000000 (PCA) | 0.993137 | invalid: queried channel was exposed |
| `reachability_auprc` | 0.933514 (PCA) | 0.910854 | invalid: sampling was trivially separable |
| `geodesic_nmae` | 0.022515 (frequency) | 0.048164 | invalid: target was nearly constant |

The valid measured anchors were `boundary_f1` (floor 0.694197, ceiling
0.887859) and `clearance_nmae` (floor 0.201552, ceiling 0.090382). The
supervised reference effective-rank fraction was 0.010898, which is collapse
evidence and is retained in `p0c-report.json`.

The run used 0.116064 accelerator-hours, within its 2.0 hour cap. P0D and P1
were not started because the frozen contract requires all P0C denominator
gates to pass first. Issue #322 selected a calibration amendment; issue #329
implements that replacement as the `voxel-encoder-pilot-v2r1` identity family.

`p0c-report.json` is the compact machine-readable record. Raw predictions,
corpora, trial stores, and model weights remain untracked.
