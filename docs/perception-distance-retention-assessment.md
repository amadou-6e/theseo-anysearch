# Distance retention after encoder retraining

Issue #368. Disposition retain. This follows the completed encoder experiment,
not a request to restart or alter it. No integration merges or promotion.

## Completed result

All three newly trained encoders retain and improve both measured distance tasks
on fresh development geometries, with no encoder updates during this validation.

| Task | Old seed scores | New seed scores | New-minus-old error 95% CI |
| --- | --- | --- | --- |
| Clearance NMAE | 0.0487 /0.0492 /0.0512 | 0.0308 /0.0320 /0.0361 | [-0.01856,-0.01475] |
| Recovery NMAE | 0.0485 /0.0503 /0.0504 | 0.0309 /0.0324 /0.0360 | [-0.01866,-0.01455] |

Lower is better. Both tasks pass the predeclared 0.02 non-inferiority margin and
absolute error bars in every seed; the paired intervals additionally favor the new
encoders. This is not merely a tolerance-based pass hiding a distance regression.

Combined with [the surface experiment](perception-encoder-continuation-assessment.md),
new encoders have boundary F1 0.733-0.738 versus old matched-probe 0.613-0.634,
and improve these two distance tasks. We have advanced to a stronger actual frozen
encoder candidate, not only a better supervised reference.

## Direction and limits

Retain the new pretraining bundle as the working local-geometry candidate. Next
validation is new-family transfer before radius or architecture scaling. Do not
claim that larger models are necessary, or that this is an efficiency-controlled
result: architecture and probe exposure were fixed, but the old/new pretraining
data and compute histories differ. Dynamic masking was already used by old LG2.

These are fresh instances of known synthetic families, not unseen-family or real
data. Occupancy retention was not separately assessed here. No topology, planning,
P1-P8 certification, global-latent, or radius-scaling claim follows.

## Audit

Source `577da2e`; spec `762262d657b4320dbce39d3b53bc028105a2896b`. Duration 21.72s.
396 garden tests passed before the evidence-only replay test was added.
[Frozen manifest](perception-encoder-local-geometry/retention-preregistration.json).
[Report](perception-encoder-local-geometry/retention-report.json).
Payload `01c0be1f42549c7c6a4638638e8a5a48120f3f3daee1c0d6a41098ad119bf577`.
Twelve probe fits; all six encoder file/state hashes verified and unchanged.
Artifacts remain untracked under runtime/distance-retention-v1-run1.
