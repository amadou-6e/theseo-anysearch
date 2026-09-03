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

---

# Reachability denominator: v2r2 successor study (#339 exploratory; v2r2 to follow)

**Status of v2r1:** terminal. P1 was recorded `not_started` (blocked P0D, blocked
amended P0C on `reachability_auprc`). That line does not reopen. A config change
cannot revive it.

**What this section is:** the amended P0C is blocked on one denominator,
`reachability_auprc` (`floor ~= ceiling ~= 0.93`), because connectivity over a
bounded, fully-visible window is near-linearly decodable from raw occupancy. The
right response is a **new partially-observed-topology task** with its own
preregistration and its own dataset / query / run identities (`v2r2`), not a
metric tweak on `v2r1`.

## #339 disposition (exploratory infrastructure only)

`exp/339` carries reusable scaffolding, not a finding:

- `pilots/reachability_fixtures.py` - graded-occlusion fixture generator;
- `evaluation/reachability_variants.py` - occlusion-span pair sampler, raw /
  rich / null feature builders, per-cell component maps;
- `experiments/perception_encoder/reachability_redesign_screen.py` - screening
  harness.

**The screen's reported gaps are NON-EVIDENTIAL.** They used a *linear* ridge
probe on raw occupancy as the floor; connectivity is nonlinear, so that floor is
too weak and every gap (A `+0.314`, B2 `+0.606`, ...) is optimistic. **Do not
adopt B2 or any variant on those numbers.** The screen established only that
occlusion *can* create headroom in principle.

Disposition: `retain` (keep the infrastructure; the result JSON stays as a
recorded non-evidential exploratory artifact). Not `promote`.

## v2r2 study design

### R0 - feasibility audit (gates everything)

Exact identical-visible-input observations are rare, so **generate controlled
counterfactual pairs**: fix the visible voxels, vary the hidden completion, and
estimate the conditional uncertainty of the completed-reachability label given
the visible input.

- unknown-channel prevalence and per-stratum class balance;
- conditional label entropy `H(reachable | visible input)` per occlusion span
  stratum. Effectively random hidden geometry -> no encoder can solve it ->
  defer. Deterministic hidden geometry -> check for generator-cue leakage
  (predict hidden structure from visible context across a held-out generator
  config);
- per-stratum floor/ceiling estimates using the R2 control ladder, not a linear
  probe.

Exit: R0 states, from calibration data only, whether a preregistered denominator
with `>= 0.10` headroom is achievable. If not, stop at the D branch below.

### R1 - occlusion-aware sampler

Completed-geometry reachability labels. Occlusion strata (`1-2`, `3-5`, `6+`)
**frozen before results**. Stratum weights **frozen before results** and
**balanced or task-motivated**, never set to observed prevalence (easy cases
would dominate again). No `max-over-strata`.

### R2 - control ladder (probe capacity frozen first)

Freeze the candidate probe capacity, optimization budget, and checkpoint-
selection procedure **before** building any control. Then compare against three
distinct controls:

1. identical probe over raw input;
2. capacity-matched **nonlinear** raw-input model;
3. over-capacity / model-free reference ceiling.

The gated `reachability` denominator is `reference - nonlinear_raw` (or, per R4,
its normalized-log-loss form), never `reference - linear_raw`.

### R3 - structural evaluation (anchor-based)

Per-cell component IDs are permutation- and count-dependent. Use **dense
reachability fields from preregistered anchor cells** (K fixed anchors per
observation by a frozen rule; target = "reachable from anchor k"). Evaluate with
VI / Adapted Rand / Betti-0 plus explicit **false-merge** and **false-split**
rates.

If `reachability` stays in the gate, the **false-merge rate is a veto**
(a false merge is a false-open). Calibrate that veto threshold from the control /
reference behaviour on calibration data - not an arbitrary absolute value.

### R4 - preregistration amendment (choose one primary metric)

From calibration data only, before P1 opens, freeze exactly one primary
`reachability` metric and its aggregation:

- normalized conditional log-loss gain
  `(null_log_loss - candidate_log_loss) / (null_log_loss - reference_log_loss)`,
  bounded and structurally uniform with the other anchors; or
- occlusion-stratified AUPRC with the frozen weighted aggregate.

AUPRC and PVI / selectivity are retained as **diagnostics only**.

The amendment must also **explicitly define the minimum active gate set** (see
open decision below).

### R5 - fresh run

New `v2r2` dataset, query, preregistration, and run identities. **No reuse of any
`v2r1` identity.** New spec commit.

### Stop rule

If R0, or R2's ladder, shows the preregistered `reachability` denominator stays
below `0.10` headroom, defer `reachability` - do **not** pick whichever
diagnostic happened to pass.

## Open decision (spec owner)

`geodesic_nmae` is already deferred. If `reachability` also defers, the active
gate is `{occupied_iou, boundary_f1, clearance_nmae}` - two of which are the
pinned calibration templates - and the pilot has **no topology-perception
coverage**.

A three-component pilot without either topology target **must not run as the
original direction-finding pilot**. The choices are:

1. **Terminate** the perception-encoder direction-finding pilot; topology
   perception is unvalidated at pilot scale.
2. Run a deliberately narrower **"local geometry only"** study under a separate,
   explicit claim (it validates local geometry representation, not topology
   perception) with its own acceptance criteria.

This is not a runtime `proceed on N-of-5`. It is a spec decision that must be
recorded before v2r2 executes.

## Tracking

- `#339` (this branch): exploratory infrastructure, `retain`.
- new **specs** issue: v2r2 partially-observed-topology preregistration
  (R0-R4 design, minimum active set, spec commit).
- new **implementation** issue: v2r2 execution (R0 audit -> R5 run) against the
  frozen v2r2 spec.
