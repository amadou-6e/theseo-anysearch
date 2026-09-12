# Compact representation diagnosis

Issue #391; source91de70a. Spec2a17aa8e0f5a92e36e3d5bb4aad1d926dc000792
(specs#64 / PR65). Run compact-search-diagnosis-v1-run1. Post-hoc explanation only,
using original probe/selection data; no development access or weight fitting.

| Checkpoint | Covariance entropy rank | Leading variance fraction | Ridge boundary F1 | Trained decoder F1 | Shuffled-vector F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid64 /512 | 2.072 | .7792 | .2693 | .2581 | .1353 |
| Grid192 /512 | 1.442 | .9161 | .2705 | .2682 | .1687 |
| Grid64 /2048 | 1.517 | .9011 | .2766 | .2658 | .1389 |
| Grid192 /2048 | 1.589 | .8680 | .2794 | .2673 | .1556 |
| Grid64 /8192 | 1.328 | .9332 | .2815 | .2811 | .1613 |
| Grid192 /8192 | 1.000001 | .99999993 | .1985 | .2034 | .1938 |

The no-input position-dependent prior scores .1902 boundary F1, .5097 IoU and
.1224/.1219 distance errors on selection. The long grid192 checkpoint is close to
this weak behavior and barely responds to shuffling. Grid64 still uses scene
information, but both its trained decoder and independent readout remain weak.
Readout replacement alone is therefore not the leading supported explanation.

Entropy effective rank describes variance concentration, not literal mathematical
rank, information bits or proof that every low-variance direction is useless.
The ablations and weak direct-decoder scores are complementary evidence. Do not
claim a causal mechanism or independent validation from this post-hoc analysis.

## Next training direction

Keep the field fixed. Test lower/decayed learning rates, explicit feature variance
and covariance regularization, and finer ordered pooling under a fresh protocol.
Use a sufficiently large feature batch for covariance estimates and retain an
unregularized control. Measure task performance alongside rank: increasing rank
alone is not success. Include an explicit control for the near-filled central
sphere/box crops before accepting pooled quality. Preserve the old search unchanged.

Variance/covariance regularization is motivated by
[VICReg](https://arxiv.org/abs/2105.04906), which studies explicit variance and
redundancy control. Applying those penalties to this supervised reconstruction
objective is an experimental adaptation, not a replication of VICReg or a claim
that its image-transfer results establish voxel success.

## Evidence and budget

All six input checkpoint hashes and encoder states verified. Regenerated original
ridge predictions matched saved predictions at rtol1e-5/atol1e-6; encoder states
were unchanged after analysis. Five focused tests passed. Report payload:
49f6cfd3cab1b0ff1c2542cdc709731c89cd6d798af4d4987fbf0999023e2e6d.
Analysis took29.297s, plus120s conservative verification/report overhead charged
to the campaign. No active process remains for this analysis. Runtime artifacts:
runtime/compact-search-diagnosis-v1-run1. Registration/report are under
docs/perception-encoder-local-geometry/compact-diagnosis-{preregistration,report}.json.
Disposition retain; no merges, promotion or usable-checkpoint claim. Continue the
fresh anti-collapse/detail training comparison within the existing48-hour budget.
