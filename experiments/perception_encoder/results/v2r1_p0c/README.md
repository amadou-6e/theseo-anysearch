# V2r1 P0C calibration result

Run `voxel-encoder-pilot-v2r1-p0c-1` completed under specification commit
`0c9e3c633799f5d42b7a603e0845cac0bd494cda` and integration base
`d9f16bf999996e1df30ea76137dffc13de4aa416`. It returned `blocked` with
disposition `reject`.

| Component | Strongest floor | Model-free ceiling | PVI gain | Result |
| --- | ---: | ---: | ---: | --- |
| `occupied_iou` | 0.597358 (PCA) | 0.955132 | 0.919173 | pass |
| `boundary_f1` | 0.591321 (PCA) | 0.986357 | 0.557548 | pass |
| `clearance_nmae` | 0.223117 (PCA) | 0.091262 | 1.453458 | pass |
| `reachability_auprc` | 0.930922 (coordinates) | 0.997126 | 0.214755 | blocked: 0.066204 headroom is below 0.10 |
| `geodesic_nmae` | 0.022515 (v2 frequency) | 0.048164 | n/a | deferred to Stage 2 |

The empirical reachability veto is 0.290387 false-open rate, derived from the
best calibrated baseline rate of 0.310387 minus the frozen 0.02 margin. The
calibration set contained 5,857 stratified positives, 3,214 component
negatives, 812 boundary positives, and 117 boundary negatives.

The run evaluated the frozen budgets of 100,000/20,000 train/calibration
coordinate queries and 50,000/10,000 train/calibration pair queries. It used
0 accelerator-hours and 0.103896 wall-hours, within the 2-hour cap. The raw
feature bank remains untracked at the content hash recorded in
`p0c-report.json`; no weights or raw corpus are committed.

The report payload SHA-256 is
`2564db175a1880cf81da40bd5a32e472f9b3db81bce111283fe48c74d409697b`.
P0D and replacement P1 cannot start under this contract because P0C did not
freeze a valid set of active denominators.
