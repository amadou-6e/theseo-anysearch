# Compact encoder C0-C3 first-increment assessment

Issue #379; source d34e534. Governing smoke spec
amadou-6e/specs@04673aba3906a50c9a51e4631ef910f8a3e767e7,
projects/theseo-anysearch/python/perception-encoder-compact-smoke.md.
Parent campaign plan4df18c69ee81c2c2ebd383b7eab615ae7583a6b3.
Authorized unmerged stack on #376. Disposition retain engineering infrastructure.
No model selection, certification, promotion or integration merge.

## Implemented and exercised

- Actual33-cubed input to compact64/128/192 vector, using ordered grid, strided or
  position-aware attention aggregation; the old global projection is unused.
- Strict query-head boundary: only compact vector plus normalized coordinate.
- Frozen-backbone and joint-training paths; fresh probes after discarding training
  heads; zeroed/shuffled fixed-head development diagnostics.
- Fresh train/probe/development parents, target/query hashes and mask checks.
-18 profiles (three aggregators, three sizes, batch1/4); six128-dimensional training
  paths at96 updates plus fresh four-task probes at128 updates each.

V1 stopped at joint CUDA backward because adaptive_avg_pool3d has no deterministic
backward. Its partial artifact and original manifest remain archived, not evidence.
V2 uses explicit separable adaptive-bin means with numerical/gradient equivalence
tests, preserving deterministic execution and renewing data/source/spec identities.

## V2 measurements

Completed in43.766 seconds including manifest preparation; data regeneration1.031s.
Single RTX3060 Ti8GB, float32; no mixed precision/concurrency/caching optimization.
Profiles include host transfer, warmup3/repeats10; local short-window timing with
coarse/noisy values, not production throughput or a reliable aggregator ranking.
Batch1 peak allocation51.8-52.5 MiB; batch4 108.3-109.0 MiB.

|128-dimensional aggregator|Parameters|96-step frozen training seconds|Joint seconds|
|---|---:|---:|---:|
|Ordered grid|144640|2.406|7.657|
|Strided|128048|2.531|7.906|
|Attention|25152|2.547|7.750|

Fresh four-task probe fitting took0.594-0.734 seconds per configuration. Budget
extrapolation to longer/data-rich runs is not yet justified; profiling must cover
steady-state throughput and validation cost before allocating the search.

## Quality interpretation

These tiny development-only runs are not useful encoder checkpoints. Five of six
configurations scored zero occupied IoU and boundary F1 at the fixed0.5 threshold.
Grid-joint scored IoU0.152/F10.138. Distance errors span about0.106-0.163 and some
zeroed/shuffled controls do as well or better. No candidate can be selected from
this smoke. Zero thresholded metrics do not by themselves identify whether the
cause is class imbalance, undertrained probes, representation loss or calibration.

Next investigate training/probe learning curves, probability distributions and
class support with calibrated controls on development/calibration data. Implement
training-only balancing and selection thresholds through a new contract, not by
retuning this report. Establish bottleneck information retention before allocating
the full search. No final test set was accessed and no48 GPU-hour search launched.

## Remaining campaign foundations

C0-C3 are PARTIAL, not complete: richer geometry and corruption suites, explicit
raw/random/spatial controls, support-aware calibration, linear versus nonlinear
probe checks, batch/worker/cache/precision profiling and complete data/API contracts
remain. C4 must still freeze numeric gates, ranking rules, run budgets and authorized
compute ceiling. Current input tests cover hidden occupancy isolation, not every
production invalid-cell configuration. Packaging and HPO remain future work.

Report docs/perception-encoder-local-geometry/compact-smoke-v2-report.json,
payload SHA256988ea76cf01b44252cfe3a6d3fc1ca87e2436486de36978a03d4330850eb5d61.
Both v1 and v2 manifests committed; six v2 checkpoint artifacts remain outside Git
with file/state hashes. Source/probe freeze integrity checked during execution.
Validation:436 garden tests passed before the added report-integrity test; all18
focused compact tests pass afterward. Regenerated data identities and all six
checkpoint file/state hashes verified. Disposable probe states/predictions were
not retained in this engineering smoke, so exact score replay requires rerunning
the pinned harness; production search must retain those artifacts too.
