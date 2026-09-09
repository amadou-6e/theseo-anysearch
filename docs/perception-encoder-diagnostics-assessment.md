# D0-D2 boundary diagnostic assessment

Issue #360. Disposition: retain. No promotion, no historical verdict changes.

## Result

The campaign completed in 74.86 seconds of recorded elapsed time including manifest
preparation, on RTX 3060 Ti / PyTorch 2.13.0+cu126, float32. D0 passed all three
seeds at F1 1.0. D1 and D2 completed every registered fit, but both missed the
engineering boundary-F1 bar of 0.70. The router returns `reference_unresolved`,
with D3 as a proposed next experiment, not an instruction to stop all research.

| Method | Seed 0 F1 | Seed 1 F1 | Seed 2 F1 |
| --- | --- | --- | --- |
| D1 supervised spatial reference | 0.3998 | 0.4373 | 0.4219 |
| D2 linear frozen-feature probe | 0.4875 | 0.5954 | 0.4317 |
| D2 32-unit frozen-feature probe | 0.5838 | 0.6018 | 0.6086 |
| D2 128-unit frozen-feature probe | 0.6005 | 0.6032 | 0.6160 |

F1 thresholds were selected on the selection split only. Every seed selected
the 128-unit head by selection log loss, not by the assessment values above.
Its mean F1 is 0.6066. At the fixed 0.5 threshold these same heads scored
0.5646/0.5626/0.5418. These are fresh development instances, not LG3 evaluation
data; differences from LG3 are not a controlled estimate of improvement.

The stratified geometry-cluster-bootstrap 95% CI for null-minus-model log loss
is [0.1289, 0.1568] for D2, but [-0.2776, -0.2284] for D1. Thus D2 extracts useful
boundary signal even though it misses the bar. D1 is worse than a train-prevalence
constant in log loss and is not a usable ceiling on encoder performance.

Supplemental D2 null-minus-model log-loss 95% intervals by family are random field
[0.1814, 0.2124], oblique sheets [0.0101, 0.1133], height field [0.1159, 0.1308],
and curved tubes [0.1701, 0.2011]. These are exploratory, unadjusted intervals,
not additional gates: 2000 geometry-cluster draws stratified by density, averaged
over all seeds, NumPy generator seed 358 advanced in the listed family order.

## What the curves tell us

D1's final training BCE is about 0.0001-0.0002. Selection loss rises from
0.314-0.367 at update 257 to 0.598-0.638 at update 2048. This is direct evidence
of severe train/selection overfitting under the fixed-data, final-step recipe.
The protocol selected final-step weights in advance; no earlier checkpoint was
substituted after observing assessment results.

The larger nonlinear probes have lower assessment log loss than the linear probes
for every seed. Threshold calibration also materially changes F1. This supports
readout sensitivity, not the claim that all missing surface information is in the
representation. Oblique sheets remain difficult (selected-head seed 2 F1 0.486).

## Next action, not another generic stop

The frozen router proposes D3 because the visible-input reference did not establish
headroom. Before attributing this to missing observations or voxel resolution,
the next scoped intervention should repair the reference's demonstrated overfitting:
compare final-step versus selection-selected checkpoints under matched compute,
on fresh development identities. Keep the encoder and observation design fixed.
This is a proposed refinement of the D3 branch, not an executed or silently adopted
protocol change. The next registration must specify it before fitting.

Do not scale the encoder or claim unidentifiability from this result. The evidence
currently implicates the evaluation recipe as well as possible representation
limitations. No universal quality claim follows from the hand-chosen 0.70 bar.

## Audit and artifacts

- Spec: `66279ceab743af9c350107fe26fca730d2b245fe` in amadou-6e/specs.
- Executable: `ebb0e44` on exp/360, stacking unmerged exp/356 and exp/358 with
  the recorded user ordering exception. Nothing merged into the integration branch.
- Registration: [frozen manifest](perception-encoder-local-geometry/diagnostics-preregistration.json).
- Full compact evidence: [report](perception-encoder-local-geometry/diagnostics-report.json).
- Report payload SHA256: `f3e74cce9bad0aebde02f24f15e082a2f7e6ad9c7261cba71be3c7bfe38b51d9`.
- Three reference fits, nine probe fits, three tiny D0 fits; zero encoder updates.
- 48 train / 24 selection / 48 assessment geometries; content and query hashes
  frozen before fitting. Hidden-target isolation and unchanged encoder hashes verified.
- Model/probe artifacts remain untracked in `runtime/encoder-diagnostics-v1-run1`;
  their SHA256 values are recorded in the compact report. No corpora or weights committed.

This binding measured boundary prediction only. It did not rerun occupancy,
distance, topology, planning, wider context or radius-scaling experiments.
