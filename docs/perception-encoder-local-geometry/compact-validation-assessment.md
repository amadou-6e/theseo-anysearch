# Frozen compact encoder: robustness and collision transfer

Issue405 / PR406; specs78 / PR79. User-authorized stack on package403/PR404.
Disposition: retain experimental evidence; no promotion or merging authorized.
Fixed input33, central17, frozen729 spatial code; no encoder retraining.

## Protocol and evidence

- Frozen specification: `191f883a24b67afe75a5ce9b0783f53351802423` in
  `amadou-6e/specs`, `perception-encoder-compact-robustness-transfer.md`.
- Frozen implementation: `e16fbc6` on `exp/405`; manifests committed as `a40a7ec`.
- Package manifest: `2ddf188ecedff4699dba434fec05a851c5649119c07dcf30c9d9afd1292d4ed1`.
- Robustness report: `6f3036bd4f6a83443324ca44a7ac9c234b82748a31b89c71be4052349de09b7e`.
- Robustness runtime artifacts: `runtime/compact-robustness-v1-run1`.
- Transfer runtime artifacts: `runtime/compact-collision-transfer-v1-run1`.

Both datasets/protocols were frozen before either study executed. The same three
packaged seeds401/402/403 are evaluated; none is selected using these results.
Weights/corpora/prediction tensors are runtime-only. Compact JSON reports and replay
evidence accompany this assessment when each study completes.

## Robustness result

Completed2016 paired observations across seven generators and twelve conditions,
with three frozen encoders/native geometry heads. All252 prediction groups and
artifacts replayed on the original CUDA path; all encoder/head state hashes match.

All four original generators pass all four original targets at baseline for all
three seeds. Among unseen generators, capsule forests pass baseline; ellipsoids
miss boundary F1 for seed401 (.6954 vs .70). Gyroid walls miss hidden occupancy IoU
for all seeds (.4513-.5363 vs .60), with boundary failures for401/403 as well.

| Condition | Seed/family groups failing any available original target (out of21) |
| --- | ---: |
| Baseline (.16 density, .20 unknown, no noise) | 4 |
| Density .03 | 21 |
| Density .08 | 18 |
| Density .28 | 6 |
| Density .45 | 18 |
| Missing .10 | 3 |
| Missing .40 | 11 |
| Missing .60 | 19 |
| Visible bit-flip noise .01 | 19 |
| Visible bit-flip noise .05 | 21 |
| Seven-slice contiguous unknown slab | 21 |

Zero missing cells leave hidden occupancy/boundary/recovery metrics unavailable
for all21 groups. Available clearance passes, but this is **not** a full-target pass.

The slab is the strongest observed failure: hidden IoU .0379-.1494 and hidden
boundary F1 .0602-.2198. At5% noise, hidden F1 .4366-.6895 misses the unchanged .70
bar in every group. At density .03, hidden IoU .0278-.5505 misses .60 throughout.
Good occupancy alone is insufficient: dense solids can retain high IoU while their
boundary F1 deteriorates. These results do not support broad distributional robustness.

The report includes support counts, per-geometry scores, all-cell diagnostics, and
231 paired mean negative-distance-error comparisons with1000 geometry bootstrap
resamples. Conditions share scene fields; density shifts change truth and are
paired-field comparisons, not repeated measurements of identical occupancy.
There are24 independent scene fields per family, not2016 independent parents.

Runtime accounting1321.922s includes600s validation reserve, freeze preparation,
and a conservative interrupted-phase charge. One Windows file-lock failure during
atomic progress replacement resumed without source/protocol changes. Original
device verification took24.735s, within the reserved validation allowance.

## Collision transfer

In progress under its independently frozen protocol. Twelve heads,4096 updates
each: compact/raw/spatial/null for seeds401/402/403. All twelve calibration thresholds
must lock before ID/OOD test predictions. Fixed-head shuffled/zeroed controls do
not refit or recalibrate. The encoder and normalization remain frozen.

No transfer or safety claim is made until the complete comparison and replay finish.
