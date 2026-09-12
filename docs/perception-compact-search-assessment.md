# Compact structural and training search

Issue #389 / PR #390. Status: search completed and artifact/metric replay verified.
Source 8d8f5f6; frozen manifest 4a804f7.
Governing spec: amadou-6e/specs@31beb05c0c1dd8fe57bfa90d6052867617bc5fdd,
projects/theseo-anysearch/python/perception-encoder-compact-search.md.
Run compact-search-v1-run1. Authorized unmerged stack on #386; no merges or
promotion authorized.

## Question and method

Compare 18 structural cells (grid/strided/attention, 64/128/192 numbers, frozen/joint
backbone), then 18 frozen training-setting templates on the top two structures.
Extend selected and protected candidates to 2048 updates, then two to 8192.
Selection alone determines the final winner before development is opened.
Independent dense ridge readouts discard the training decoder. This does not
qualify an independently trained small coordinate-query head.

## Interpretation limits

- Pooled occupancy results can be dominated by the sphere/box central regions.
  Early selection scores around .74 coexist with near-zero sheet/random-field
  IoU. Keep family results visible; pooled targets do not establish broad usability.
- The grid mode projects through 128 hidden activations, and attention concatenates
  four 32-wide queries. Their 192-number outputs therefore do not test 192
  independent linear feature directions. This is the frozen architecture, not a
  post-hoc source change; output length is not equivalent to effective capacity.
- Training curves log individual minibatches, not a fixed evaluation set. A lower
  logged loss alone is not evidence of better representation or generalization.
- All runs share one initializer/sample seed. The synthetic four-family development
  set is not a final independent qualification suite. New matched raw/random/spatial
  controls are still required before claiming pretraining benefit.
- Dense ridge readouts contain (D+1)*14739 parameters; they are not small query
heads. No local feature or raw-input bypass enters those readouts.

## Completed result

All 36 configurations and 46 training stages completed. The selection-only winner
is trial 0: grid pooling, 64 numbers, frozen backbone, 8192 updates, AdamW learning
rate .003 and weight decay .01. Neither finalist is qualified.

| Finalist | Development IoU | Boundary F1 | Clearance NMAE | Recovery NMAE |
| --- | ---: | ---: | ---: | ---: |
| Grid64 frozen, selected winner | .6979 | .2756 | .09636 | .09607 |
| Grid192 frozen | .5130 | .2052 | .11323 | .11315 |
| Frozen directional target | .60 | .70 | .10 | .10 |

For the winner, family occupancy IoU is .0438 random field, .0632 sheets, .7773
spheres and .8798 boxes. Boundary F1 is .2546/.3396/.2195/.1955 respectively.
Distance errors on sphere/box families remain around .17/.18, despite pooled
errors below .10. This is not broad geometry retention or a usable frozen encoder.
Disposition: retain the search infrastructure and negative evidence; no promotion.

Longer training is not uniformly helpful: grid192 selection boundary F1 went from
.27945 at 2048 updates to .19849 at 8192, with occupancy IoU falling from .7252 to
.5097. Grid64 selection boundary F1 improved only from .27662 to .28147. The
protected joint grid192 run also deteriorated at 2048. Hyperparameter tuning did
not overcome the boundary weakness in this search space. Do not conflate this with
proof that every compact encoder or every nonlinear readout must fail.

Next bounded action is to diagnose readout versus representation limitations:
compare trained-decoder behavior with fresh frozen readouts, measure embedding
rank/necessity, and use controls to distinguish spatial priors from retained detail.
Use these observations to direct a fresh training protocol; do not silently replace
the selected winner, lower targets or turn this development set into a final test.

## Durable checkpoint

User explicitly authorized up to 48 GPU-hours on 2026-09-12. The campaign ledger
is runtime/compact-campaign-budget.json. One historical hour is conservatively
reserved; the first search tranche reserves up to eight hours including CPU phases.
The interruption left trial 0 complete and trial 1 unfinished. On resume, completed
trial 0 was retained and 900 seconds conservatively charged for the interrupted
stage, following the frozen protocol. Runtime progress and artifacts remain in
runtime/compact-search-v1-run1. Process session 21140 completed successfully.

The ledger settled this tranche at 2530.155 seconds, including the conservative
interruption charge. Together with the historical reserve, 6130.155 seconds are
accounted for, leaving approximately 46.30 authorized hours. This is conservative
charged elapsed time, not an assertion that the GPU was busy for every second.
Post-run verification/diagnosis overhead is recorded separately, not silently free.

Next action: publish the completed search evidence and execute the bounded diagnosis
within the remaining budget.
Do not infer a qualified encoder from intermediate metrics or restart completed
trials. No weights or generated corpora belong in Git.

Validation: 455 garden tests passed; two additional verifier
tests passed. The verifier refuses incomplete or hash-tampered reports before data
access. It verified all 48 artifact hashes and exactly replayed 46 selection-stage
metrics/thresholds and both development assessments. This checks saved predictions,
not reproduction of training. Report payload SHA256:
572e8910c6d1934f59662a9b79312f0a08e482fb91dfe2d832f08a6bca8048a6.
Compact report: perception-encoder-local-geometry/compact-search-report.json.
