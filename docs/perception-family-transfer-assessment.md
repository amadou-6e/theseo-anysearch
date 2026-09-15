# Frozen encoder new-family validation

Issue #370. Disposition retain. Outcome `transfer_supported` under the registered
two-family protocol. No promotion, merges, or changes to earlier verdicts.

## Results

All three seeds passed all four task checks on both held-out generator families.
Ranges below are across seeds, not confidence intervals.

| Held-out family | Occupancy IoU | Boundary F1 | Clearance NMAE | Recovery NMAE |
| --- | --- | --- | --- | --- |
| Shells | 0.869-0.878 | 0.783-0.805 | 0.0366-0.0442 | 0.0366-0.0431 |
| Trigonometric labyrinth | 0.793-0.810 | 0.769-0.781 | 0.0302-0.0400 | 0.0296-0.0393 |

Higher is better for IoU/F1; lower for distance errors. The old encoders' boundary
F1 ranges were 0.683-0.689 on shells and 0.677-0.704 on labyrinths. All eight
paired geometry-bootstrap improvement intervals favor the new encoders, not only
the non-inferiority margins. These are pointwise, unadjusted intervals, reported
in full in the machine-readable evidence.

## What was held out

Encoders remained unchanged. Each task used a fixed 32-hidden probe trained on
192 geometries from the original four generator families. Classification thresholds
were selected on 48 additional original-family geometries. Only assessment used
shells and trigonometric labyrinths, 48 geometries total balanced by family/density.
There was no fitting, capacity selection, threshold tuning or checkpoint selection
on these new families. This differs from earlier target-domain-adapted evaluations.

## Direction

The new local-feature encoders now have supporting evidence for surface recovery,
distance retention, and transfer to these two held-out procedural families. Retain
them as the working candidate for the next scale/context experiment. The evidence
does not require an architecture change before testing scale.

This is not universal generalization: all data remain synthetic, 17-cubed crops,
the same missingness mechanism and occupied-density range. Previous oblique-sheet
weaknesses are not erased by success on these families. No topology, planning,
global embedding, real-world or radius-scaling validation has occurred. The old/new
pretraining histories differ in data and compute, so this is not an efficiency-
controlled architecture comparison. No P1-P8 topology-program gate is reopened.

## Audit

- Executable source `1a73724`; frozen spec `bc3a092d07309eb8f22390bd2878f371713badad`.
- [Registration](perception-encoder-local-geometry/transfer-preregistration.json).
- [Full report](perception-encoder-local-geometry/transfer-report.json).
- Payload `f48a2741debf123070493623f078fd8832cc6900bcd9855fe54f07669f26124c`.
- 24 probe fits, six frozen encoders, zero encoder updates; recorded duration 37.94s.
- Full garden suite: 401 passed before evidence-only replay test addition.
- All 24 artifact hashes and exact CUDA prediction-to-statistic replays verified.
  CPU distance reductions differ by at most 1.91e-6 in summed errors; the original
  CUDA reduction path reproduces the stored statistics exactly.
- Weights/predictions remain untracked in `runtime/family-transfer-v1-run1`;
  hashes recorded in the report. No generated corpus committed.
