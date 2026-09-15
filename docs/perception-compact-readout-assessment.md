# Independent nonlinear compact readouts

Issue #395 / PR #396. Sourced02d086, manifest11811c9.
Speca958699f38e83ee43d4673d442591a65d34bbc72 (specs#68 / PR69).
Run compact-readout-v1-run1 completed20 fits and34 checkpoint evaluations. Encoders
remained immutable; no raw/local-feature bypass entered candidate heads.

## Learnability and real-code results

The convolutional head passed all three four-scene overfit controls: occupancy
IoU1.0, boundary F1.989/1.0/1.0 for LR.0003/.001/.003. FiLM passed at.003 with
both scores1.0, but underfit at.0003 (.584/.700) and.001 (.878/.915). These are
same-scene controls, not generalization or real-encoder scores.

All12 real frozen-code head configurations remained weak. Selection boundary F1
was roughly.34-.35, despite successful learnability controls. The higher-diversity
encoder did not solve the problem. Independently fitted no-input heads scored
about.178 IoU/.259 boundary F1; ridge controls scored boundary F1.3297/.3288.
The real codes carry useful signal above these controls, but not enough quality.

| Locked development finalist | IoU | Boundary F1 | Clearance NMAE | Recovery NMAE |
| --- | ---: | ---: | ---: | ---: |
| Source0, conv, LR.001,4096 steps (selected) | .25684 | .33722 | .09771 | .09790 |
| Source0, conv, LR.003,1024 steps | .25616 | .33690 | .09911 | .09905 |

Both fail classification targets. Selected-head family boundary F1:
random.2628, sheets.3426, spheres.3388, boxes.3938. Sphere/box distance errors
remain about.12. No usable-encoder, certification or promotion claim is justified.
Disposition retain the head implementations and evidence, not these encoder weights
as a successful compact perception solution.

## Direction

Stop allocating additional search to heads on these fixed codes. The next bounded
comparison should train the aggregation with a nonlinear reconstruction decoder,
preserve finer spatial detail and compare genuine64/128 capacity without a smaller
hidden linear bottleneck. Reuse the existing GroupNorm preactivation residual block
for a learned spatial aggregation candidate. Include a contemporaneous local spatial
reference so a compact failure is not misattributed when the observation/task or
frozen backbone is itself inadequate. Keep the33 input and central17 target field.
Use fresh identities and frozen rules; do not revise this study's winner or targets.

This is an evidence-directed engineering choice, not a theorem that all compact
encoders fail or a causal proof about dimensionality. Nonlinear heads can fit the
tiny fixtures, but that does not guarantee every complex latent representation is
easy to read with the chosen optimization budget.

## Verification and budget

All39 artifact hashes verified. Original CUDA inference reproduced all34 saved
heads, all24 shuffled-candidate results, both ridge controls and both development
metric sets. Normalization, selection-lock and source/cache identities checked.
Verifier: scripts/verify_compact_nonlinear.py. Source tests:20 focused/regression
tests plus two verifier tests passed; full garden suite484 passed in91.42s.

Report payload39503c4fdbb641581a3d1d587827e6fe30e46126dae96e46f9a191d9092fd6e8.
Runtime: runtime/compact-readout-v1-run1. Charged report elapsed1180.906s includes
measured preparation and300s overhead reserve. Ledger settlement additionally
includes final report writing. No active experiment process remains for this run.
Registration/report: docs/perception-encoder-local-geometry/compact-readout-
{preregistration,report}.json. Continue within the approved48 GPU-hour campaign;
integration merges remain unauthorized.
