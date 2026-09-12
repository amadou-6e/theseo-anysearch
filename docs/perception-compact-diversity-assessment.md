# Compact diversity and detail comparison

Issue #393 / PR #394. Sourceb09313a, manifest0f042ca.
Spec39b49f162140cfa6114f95dd8151b5bf511dc1b7 (specs#66 / PR67).
Run compact-diversity-v1-run1 completed six recipes and18 checkpoint evaluations.
Disposition retain infrastructure/evidence; no usable encoder or promotion claim.

## Result

All outputs contain64 numbers. Table shows1024-update selection results, not the
selection procedure's chosen checkpoint. Family-macro ranking selected the
unregularized recipe0 at512 updates, not the highest-rank representation.

| Recipe | Pool | Regularizer | Entropy rank | Pooled IoU | Boundary F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original-style optimizer | 5 | 0 | 10.33 | .2501 | .3291 |
| Lower cosine LR | 5 | 0 | 4.41 | .2536 | .3270 |
| Lower LR + penalties | 5 | .1 | 6.67 | .2554 | .3273 |
| Lower LR + penalties | 5 | 1 | 12.78 | .2534 | .3217 |
| Finer pool + penalties | 9 | .1 | 8.76 | .2534 | .3261 |
| Finer pool + penalties | 9 | 1 | 21.62 | .2620 | .3301 |

The largest rank improvement did not materially fix reconstruction. Rank is not a
utility metric. Strong penalties also weakened the jointly trained linear decoder.
Finer pooling plus regularization is not established as the winning recipe.

Locked winner recipe0/512 development: IoU.24850, boundary F1.33044,
clearance NMAE.09825, recovery NMAE.09837. Other finalist recipe1/1024:
.24866/.32545/.09572/.09609. Both fail classification targets. Winner family F1:
random.2513, sheets.3335, spheres.3330, boxes.3858. Sphere/box distance errors
remain about.119/.117. No family or pooled qualification claim is justified.

This corpus balances central-region density. The old sphere/box filled-crop
advantage is absent; do not interpret cross-study pooled IoU changes as an isolated
effect of the new regularizer. All new recipes had identical data and mask banks.

## Direction

Increasing feature diversity alone is insufficient. Test a nonlinear independently
trained readout before allocating more search to these penalties. Freeze the
selected task checkpoint and a high-diversity contrast checkpoint as inputs to a
fresh readout study; compare spatial convolutional and feature-conditioned query
decoders with identical label budgets and an explicit no-input control. Keep the
encoder frozen and64-dimensional. Any subsequent encoder retraining or decoder
selection must use fresh protocol/data identities, not reused development results.

## Verification and efficiency

473 garden tests passed. All21 artifact hashes,18 threshold/metric replays,
18 cache -> aggregation -> ridge prediction replays and both development metric
replays passed on the original CUDA path. The initial CPU inference replay exceeded
rtol1e-5/atol1e-6 for some near-zero outputs; maximum absolute CPU/GPU difference
was4.055e-6. No tolerance was relaxed: original-device verification passed. CPU
portability at that strict tolerance is not certified. Original weights unchanged.

Each1024-update recipe plus its evaluations took about9.3-9.7 seconds once exact-
mask features were cached. Each recipe processed131072 bank entries, not131072
unique scenes. Frozen caching removed repeated backbone computation; it does not
create independent data. Eight registered masks per training scene were used.

Tranche charged484.922s including300s overhead reserve and measured freeze
preparation. Ledger: runtime/compact-campaign-budget.json. No active process remains
for this run. Runtime: runtime/compact-diversity-v1-run1. Report payload SHA256:
02701cbde1957a598460a86b0a4321f3ff39344a3a468c0f54dd3b943787fbe0.
Registration/report: docs/perception-encoder-local-geometry/compact-diversity-
{preregistration,report}.json. Verifier: scripts/verify_compact_diversity.py.
Continue the nonlinear-readout comparison within the existing48-hour authorization;
no integration merge or larger field is authorized.
