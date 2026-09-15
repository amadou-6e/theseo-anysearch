# Fixed-architecture encoder continuation

Issue #366, PR #367. Disposition retain; no promotion or merges.

Three new encoders completed 2048 joint occupancy/ESDF updates each on independent
pretraining data. All probe comparisons used frozen encoder weights afterward.

| Seed | Old, 48 probe geometries F1 | Old, 192 F1 | New, 192 F1 | Reference F1 |
| --- | --- | --- | --- | --- |
| 0 | 0.6220 | 0.6343 | 0.7380 | 0.7596 |
| 1 | 0.5948 | 0.6129 | 0.7328 | 0.7201 |
| 2 | 0.6218 | 0.6239 | 0.7374 | 0.7488 |

Paired log-loss improvement from probe exposure alone: 0.00374, 95% CI
[0.00243,0.00525]. New versus old at matched probe exposure: 0.05418,
CI [0.04678,0.06215]. New versus reference: 0.01906,
CI [-0.00084,0.04339], so reference superiority is not resolved.

Direction: validate distance retention and new-family transfer before scaling.
The surface gain is not explained just by giving the probe more data. Architecture
was held fixed, but pretraining distribution, count, update budget and learning
rate changed together. Old LG2 already used dynamic masks; fresh masking alone
cannot explain this result. This is not a matched-pretraining-compute ablation.

No distance-retention, unseen-family, topology or radius claims yet. Those need
separate measurements; no historical verdicts are amended.

Recorded duration 336.31 seconds; source 2423714; frozen spec
4349ad2ac47cbad912a7a984407f97bbd9042628. 393 garden tests passed.
[Manifest](perception-encoder-local-geometry/continuation-preregistration.json).
[Full report](perception-encoder-local-geometry/continuation-report.json).
Report payload: 3756d04628a9c6e377e9ae67853ac440fe4afeae46ae4160ea064d8965710390.
Weights remain untracked under runtime/encoder-continuation-v1-run1.
