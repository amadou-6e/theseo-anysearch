# Perception Encoder Calibration Revision Work Plan

Status: foundation implemented; execution pending review
Integration branch: `exp/perception-encoder`
Trigger: P0C returned `blocked` — three of five denominator gates failed
(`occupied_iou`, `reachability_auprc`, `geodesic_nmae`). Root causes in
[`p0c-calibration-revision-research.md`](perception-encoder-calibration-revision-research.md).
Specifications: pilot spec at
`amadou-6e/specs@0c9e3c633799f5d42b7a603e0845cac0bd494cda`.

## Objective

Repair the probe and anchor-calibration design so P0C can pass its denominator
gates on a meaningful basis, re-run P0C/P0D, and re-run P1 under the amended
contract. Retire the v1 P1 `no_viable_direction` verdict, which rests on a
false-open veto shown by P0C to be unpassable by a fixed random projection.

Each row becomes a GitHub issue and an `exp/<issue-number>` branch. Foundation
tasks deliver tested code plus a deterministic CPU smoke path. Execution tasks
deliver locked configurations, compact reports, artifact hashes, and a
machine-readable decision. No branch merges to `develop`.

## Scope boundary

In scope: the three failed metrics, the ceiling/reference method, the score
aggregation, the preregistration amendment, and the re-runs of P0C/P0D/P1.

Out of scope: `boundary_f1` and `clearance_nmae` (calibrated cleanly — frozen as
the template), encoder architecture, pilot P3–P8 methodology, promotion to
`develop`, and any change to the four objective bundles T0–T3.

## Work packages

| ID | Work package | Primary ownership | Depends on |
|---|---|---|---|
| F0 | Amendment contract: revised-anchor + revised-probe schema, new dataset id, new run-identity scheme, and the machine-readable v1 P1 supersede record citing `p0c-report.json` `report_payload_sha256` | `garden/pilots/`, new revision section in the pilot spec | none |
| F1 | Model-free ceiling estimators: Bayes-error (kNN / MST / direct) for classification metrics, kNN-residual ceiling for regression metrics, and an effective-rank non-collapse guard; replace `calibrate_score_anchors` ceiling source | `garden/pilots/runner.py`, new `garden/evaluation/ceilings.py` | F0 |
| F2 | Triviality instrumentation: pointwise-V-information and MDL codelength, null-input baselines (zeros, coordinates-only), and a per-metric denominator pre-check that blocks a metric when `PVI(task \| null) ≈ PVI(task \| embedding)` | new `garden/evaluation/triviality.py` | F0 |
| F3 | `occupied_iou` probe redesign: masked / held-out occupancy queries (cells masked from encoder input only), off-grid query option, cross-channel variant behind a flag | `garden/pilots/comparative.py`, `garden/evaluation/probes.py` | F0, F2 |
| F4 | Reachability / false-open probe redesign: geodesic-distance-stratified pair sampling with a positive/negative separation margin, boundary-perturbation negatives, optional two-way consistency, per-distance-bin AUPRC, held-out-fold threshold calibration, and an empirically derived veto relative to the calibrated baseline distribution | `garden/pilots/comparative.py`, `garden/evaluation/metrics.py` | F0 |
| F5 | Geodesic probe: redesign (per-geometry-max normalization, stratified sampling, multi-horizon Spearman / ordinal consistency) OR a recorded Stage-2 deferral with rationale | `garden/pilots/comparative.py`, pilot spec | F0, F4 |
| F6 | Supervised-reference training-recipe fix (warmup, cosine schedule, gradient clipping, weight EMA for evaluation, evaluation cadence 20–50 updates) plus an optional multi-task trunk for an encoder-ceiling; enforce `effective_rank_fraction > 0.3` before acceptance. Conditional: only if a trained reference is retained after F1 | `garden/pilots/runner.py` | F0 |
| F7 | Scoring / aggregation revision: replace the `pilot_score` unweighted mean with per-metric reporting plus IQM and performance profiles; gate on per-metric anchors only; freeze `boundary_f1` and `clearance_nmae` as the calibration template with a documented acceptance bar | `garden/pilots/benchmark.py`, `garden/evaluation/statistics.py` | F1, F2 |
| F8 | Revised P0C contract integration and amended-preregistration freeze: wire F1–F7 into one `run_p0c` path, regenerate the v2 foundation config, and freeze the amended preregistration (new spec commit, dataset id, run ids) | `garden/pilots/runner.py`, `experiments/perception_encoder/`, pilot spec | F1–F7 |
| E1 | Execute the re-run P0C calibration under the amended contract | locked config and reports under `experiments/perception_encoder/results/v2_foundation/` | F8 |
| E2 | Execute P0D four-times-data sensitivity | locked config and reports | E1 |
| E3 | Execute the re-run P1 objective screen; record the v1 P1 verdict as superseded | locked config and reports | E2 |

## Acceptance per package

- **F0** — schemas round-trip deterministically; the supersede record validates,
  names the fired veto, and embeds the P0C report hash; invalid or incomplete
  amendment manifests fail with an explicit message; CPU unit tests cover all
  contract fixtures.
- **F1** — estimators recover known ceilings on synthetic fixtures with known
  Bayes error; the non-collapse guard rejects a rank-2 fixture; no CUDA required
  for the estimators.
- **F2** — PVI and MDL recover known values on synthetic fixtures; the pre-check
  flags a task whose label is a deterministic function of the null input and
  passes a task that is not.
- **F3** — a linear probe over PCA / fixed-random-projection of the input scores
  at or below the calibrated floor on the redesigned `occupied_iou`, not 1.0;
  masked cells are provably absent from the encoder input.
- **F4** — a fixed random projection scores below the calibrated
  `reachability_auprc` floor and above the false-open veto only by the documented
  margin; per-bin AUPRC and calibrated-threshold error rates are reported
  separately; the veto threshold is derived, not hard-coded.
- **F5** — either the frequency-baseline geodesic NMAE exceeds the documented
  noise floor and the supervised reference improves on it, or a Stage-2 deferral
  is recorded with rationale and the metric is removed from the P0C gate set.
- **F6** — the reference converges without the update-400-style divergence over
  three fixture seeds; `effective_rank_fraction > 0.3` at the selected checkpoint;
  skipped cleanly when F1 leaves no trained reference in the loop.
- **F7** — per-metric report includes IQM and a performance profile; no composite
  score can mask a single degenerate denominator; the frozen template metrics are
  unchanged and their acceptance bar is documented.
- **F8** — one deterministic CPU smoke command runs the full revised P0C path on
  fixtures; the amended preregistration writes a new spec commit, dataset id, and
  run ids, and content-addresses every artifact.
- **E1** — all five denominator gates pass, or a metric is explicitly deferred
  with a recorded reason; the ceiling method's non-collapse check passes; report
  includes accelerator-hours against the cap and a `promote` / `retain` / `reject`
  disposition following the amended rule.
- **E2** — matched-budget results with geometry-cluster uncertainty; the
  preregistered P0D decision, or `blocked` with `no_retained_bundle` if E1 did not
  pass.
- **E3** — the amended P1 decision follows the preregistered rule from the
  recorded numbers, or a logged deviation cites F0's amendment; the v1 P1 report
  is marked `superseded_by` this run's identity.

## Non-goals

- Changing `boundary_f1` or `clearance_nmae`.
- Tuning the redesigned probes until `supervised − floor ≥ 0.10`; the fix is to
  make the tasks meaningful, verified against the Bayes-error / PVI baseline.
- Widening the pilot radius, changing capacity families, or altering T0–T3.
- Promotion to `develop`.

## Parallel schedule

```text
F0
 |-- F1 model-free ceilings ------|
 |-- F2 triviality instrumentation|
 |     `-- F3 occupied_iou -------|--> F7 scoring --|
 |-- F4 reachability/false-open --|                 |--> F8 revised P0C freeze
 |     `-- F5 geodesic or defer --|                 |
 `-- F6 reference recipe (cond.) -------------------|

F8 --> E1 P0C' --> E2 P0D --> E3 P1'
```

After F0 merges, F1/F2/F4/F6 run in parallel (separate file ownership). F3
follows F2; F5 follows F4. F7 joins F1+F2; F8 is the integration join. E1–E3 are
decision-dependent and stay ordered.

## Integration gates

Every foundation PR provides unit tests and one deterministic CPU smoke path;
CUDA-only paths report an explicit skipped state. F8 cannot pass until the revised
`run_p0c` fixture path, the triviality pre-check, the ceiling non-collapse guard,
and the amendment-manifest fixtures all pass.

Every execution PR records the issue, integration-base SHA, code SHA,
dataset/query hashes, resolved config, trial counts against the cap, per-geometry
metrics by artifact reference, accelerator-hours, the machine-readable decision,
and an explicit `promote` / `retain` / `reject` recommendation. No execution
result changes a threshold after its evaluation pool is opened; changes to
preregistered behaviour require a further spec revision and a new run identity.

## Promotion boundary

After E3, open one promotion issue. Cherry-pick only the foundation commits the
accepted direction needs plus compact reports. If E1 or E3 returns
`no_viable_direction`, the promotion inventory is foundation commits only. The
promotion PR to `develop` reruns all garden unit tests and the deterministic P0C
smoke on the current development line.
