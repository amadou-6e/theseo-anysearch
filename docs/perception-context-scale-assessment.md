# Paired 17 to 33 context pilot

Issue #372, PR #373. Disposition retain. Completed all three seeds and four arms.
No integration merges, promotion or automatic 65-cubed escalation.

## Main result

Shared frozen encoder weights transfer to 33-cubed crops, with improved predictions
at the same central query cells. Probes are separately fitted at each size.
This is more context at fixed voxel resolution, not upsampling the same scene.

| Task | Frozen17 seed range | Frozen33 seed range | Paired improvement 95% CI |
| --- | --- | --- | --- |
| Occupancy IoU | 0.660-0.679 | 0.687-0.717 | [0.0177,0.0439] |
| Boundary F1 | 0.696-0.722 | 0.739-0.756 | [0.0234,0.0440] |
| Clearance NMAE | 0.0412-0.0463 | 0.0310-0.0359 | [0.0086,0.0123] |
| Recovery NMAE | 0.0402-0.0453 | 0.0305-0.0354 | [0.0077,0.0116] |

All four frozen-transfer non-inferiority checks pass, and all four improvement
intervals are positive. Classification gains mean higher scores; distance gains
mean lower errors. The aggregate applies to this two-family mixture; per-family
scores remain in the report, not replaced by a universal family-quality claim.

Matched 512-update retraining also improves at 33 relative to 17, but does not beat
the established frozen33 encoder on surfaces: retrained33 IoU 0.633-0.659 and
boundary F1 0.664-0.708. The comparison to frozen33 has negative surface-gain
intervals. Distance differences between retrained33 and frozen33 are unresolved.
This short retraining budget is not evidence against fully tuned scale training.

## Cost

For batch 4, mean incremental peak allocated inference memory was 9.93 MiB at 17
and 72.91 MiB at 33, about 7.34x. Total frozen-arm allocated peaks were 142.1 and
217.8 MiB; these include resident data/model tensors and are not total device use.
Mean frozen-arm forward timing was 13.0ms versus 15.6ms per batch. The short timing
measurements are noisy: retrained arms measured 13.3ms at both sizes. Do not infer
near-free production scaling or extrapolate these timings to larger volumes.

Mean 512-update training time was 19.3s at 17 versus 35.4s at 33; allocated peaks
170.6 versus325.7 MiB. Equal updates process more voxels at33, so these are not
equal-FLOP or equal-target-count runs. No out-of-memory failure occurred.

## Next direction

Keep the validated shared encoder weights. A separately registered 65-cubed
frozen-transfer/resource check is justified before spending on per-scale retraining
or a new architecture. Preserve fixed spacing, nested queries and parent targets;
start with memory profiling rather than extrapolating this result to radius 512.
No claim is made about the entire 33-cubed output, only central 17 query cells.

This pilot used 48 pretraining /48 probe /24 selection /48 assessment parent scenes,
two stationary families, parent 49 grids and 512-update fresh retraining. It is not
directly comparable to the larger earlier training corpus or an unseen-family test.
No topology, planning, global-latent, real-world or promotion claim follows.

## Audit

Source `6dea43e`; spec `b950249860b3e33640e9faadc3fe0a7ecabc92fc`.
Recorded duration 247.15s. Full garden suite: 406 passed before evidence-only additions.
[Manifest](perception-encoder-local-geometry/scale-preregistration.json).
[Full report](perception-encoder-local-geometry/scale-report.json).
Payload `e66dfdecf165ee75bbf6255741968d3acb3457a696f9b851a385c444af062bfe`.
Six new 512-update encoders; frozen-arm checkpoints unchanged; 48 probe fits.
Weights and predictions remain untracked under runtime/context-scale-v1-run1.
