# Compact query-decoder learnability assessment

Issue #385, source91d50ad, spec1bf1359b55b7c9aef2de1fe23a831e2e850009e0.
Run compact-learnability-v1-run1. Disposition retain diagnostic. No promotion.

All six2048-update tiny-corpus fits completed; none met IoU.95/F1.97 memorization
targets. The analytic decoder from the six generator parameters exactly matches
every target, proving the oracle code contains sufficient information.

|Source|Decoder bundle|Final IoU|Final F1|Fit seconds|
|---|---|---:|---:|---:|
|Oracle6|Plain32|.199|.333|3.391|
|Oracle6|Fourier128|.242|.389|4.375|
|Learned table128|Plain32|.200|.333|4.047|
|Learned table128|Fourier128|.374|.544|4.937|
|Voxel encoder128|Plain32|.195|.327|141.891|
|Voxel encoder128|Fourier128|.334|.500|142.625|

The plain query head cannot fit even an information-complete code under this
optimization setup. Larger/Fourier readout improves fits but remains inadequate.
This does not prove128 is too small, nor isolate coordinate features from decoder
capacity: both changed together. Scores describe four-scene memorization, including
visible voxels, not generalization or hidden-only perception quality.

Following the preregistered routing, #386 is already prepared: direct code-to4913
logits linear decoder as a positive control on four fresh scenes. Its oracle
design matrix has rank4 and can express arbitrary four-scene labels. It removes
coordinate-query decoding; matched-compute or architecture-superiority claims
will not be made. Do not launch HPO until a learnable compact path is established.

Elapsed303.141 seconds including preparation.447 garden tests passed. Source/model
states and full final predictions saved outside Git, six artifact hashes recorded.
Registration/report under docs/perception-encoder-local-geometry/
compact-learnability-{preregistration,report}.json. Payload
00e7511e6dd9ba360140428fc249689b9f7c5db0dab62b28bd226f58123ea256.
No active process remains for this run.30-minute total/10-minute per-fit caps not
exceeded. Larger search budget approval remains separate; no merges performed.
