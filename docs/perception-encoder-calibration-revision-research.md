# P0C Calibration Failure: Revision Research

Status: research input for a recorded protocol amendment
Owner: amadou-6e
Related: `experiments/perception_encoder/results/v2_foundation/p0c-report.json`
(`report_payload_sha256: a7a149f9235b38f9ff1f1a230ce791367cf528fc6705e0f987674f1a48d4ea43`, local run artifact),
issue #327, supersedes the v1 P1 null result recorded in #313
Spec commit at P0C: `01eefc529016da48c4a1dd17b85391720542af14`
Code at P0C: `880e19b56b5e27d02d8d1132b3f076125a75668a`

## 1. What P0C measured

P0C ran anchor calibration on `voxel-encoder-pilot-v2-dataset-1`
(`dataset_sha256: 0109834e…`) in 0.116 accelerator-hours (cap 2.0) and returned
`decision: blocked` — three of five measured floor/ceiling denominator gates failed.

| Metric | Floor (baseline) | Ceiling (supervised ref) | Gate | Interpretation |
|---|---|---|---|---|
| `boundary_f1` | 0.694 (fixed random projection) | 0.888 | pass | healthy: 0.19 headroom, reference on top |
| `clearance_nmae` | 0.202 (PCA) | 0.090 | pass | healthy: 55% error reduction |
| `occupied_iou` | **1.000 (PCA)** | 0.993 | **fail** | ceiling below floor |
| `reachability_auprc` | **0.934 (PCA)** | 0.911 | **fail** | ceiling below floor |
| `geodesic_nmae` | **0.0225 (frequency)** | 0.0482 | **fail** | supervised worse than a constant predictor |

Supporting detail from `runtime/perception_encoder/v2_foundation/p0c-evaluations.json` (local run artifact):

- `occupied_iou = 1.000` for **both** PCA and fixed random projection.
- `reachability_auprc`: fixed random projection 0.926, frequency 0.830.
- `false_open_rate`: PCA 0.192, fixed random projection 0.148, frequency 0.265,
  supervised 0.269. The v1 P1 hard veto is `false_open_rate > 0.05`.
- `false_closed_rate` ~ 0.31–0.35 for every method, supervised included.
- Supervised reference embedding: `effective_rank 2.16`,
  `effective_rank_fraction 0.011`, `largest_component_fraction 0.998`.
- Supervised reference geodesic curve: `selection_error` 0.14 → **1.53** → 0.14
  across updates 300–500; only 5 evaluation points over 500 updates;
  `selected_update 300`; `residual_training_error 0.156 > selection_error 0.139`.

## 2. Diagnosis — three distinct defects, not one recalibration

### A. `occupied_iou` probe is structurally degenerate (task triviality / coordinate leakage)

PCA and fixed random projection both score exactly 1.000. The coordinate probe
reads `[embedding, query_coordinates, stride_metadata]` and is asked to
reconstruct the `occupied` input channel; the query coordinates plus a near-identity
linear read of that channel already solve it. No encoder can exceed 1.0, so the
denominator is zero or negative. This is the positional-shortcut failure documented
for coordinate-conditioned 3D prediction (Jiang et al. 2023, CVPR; MPL-MAE 2026):
the probe/decoder solves the task from coordinates, not from the representation.

### B. Reachability / false-open probe is a label/threshold artifact

A fixed random projection reaches `reachability_auprc` 0.926 and `false_open_rate`
0.148; frequency reaches 0.830 / 0.265; universal `false_closed_rate` ≈ 0.31.
The v1 P1 veto `false_open_rate > 0.05` is **unpassable by a random projection of
the input** — it measured a property of the pair sampling and decision threshold,
not encoder quality. Random pair sampling makes most negatives trivially far apart,
saturating pairwise AUPRC (cf. SPTM's positive/negative separation margin M·l,
Savinov et al. 2018).

### C. `geodesic_nmae` is ambiguous between "no signal" and "bad reference"

Frequency-baseline NMAE 0.0225 means predicting the mean is already near-perfect —
the normalized target has almost no variance on this pool, so there is little to
learn. Compounding this, the supervised reference (see D) did not converge on this
head. Both causes are plausible and both must be removed before the metric is usable.

### D. The supervised reference collapsed — the ceiling is not trustworthy

`effective_rank 2.16 / fraction 0.011`, `largest_component 0.998`: the reference's
penultimate representation has collapsed to roughly two dimensions. This is
textbook Neural Collapse (Papyan, Han, Donoho 2020, PNAS): a converged supervised
classifier's penultimate features collapse to about (#classes − 1) dimensions **by
design** (3-class occupancy → ~2D). A plain supervised classifier's embedding
therefore cannot serve as an "encoder ceiling." The update-400 divergence spike
(0.14 → 1.53) is a standard optimization transient (LR too high / no warmup / no
gradient clip / eval cadence too coarse at every 100 updates).

`occupied_iou` fails on triviality regardless of D (PCA = 1.0). `reachability`
fails on both B and D. `geodesic` fails on both C and D.

## 3. Relevant sources of information

### Is a probe measuring the encoder, or is the task just easy?

- **Ethayarajh, Choi, Swayamdipta 2022, "Understanding Dataset Difficulty with
  𝒱-Usable Information"** (ICML), code `github.com/kawine/dataset_difficulty` —
  𝒱-usable information / pointwise-V-information (PVI): a principled scalar for how
  much *usable* signal input carries about a label for a model family. Compute
  under the real embedding vs a null input (zeros / coordinates-only); equal values
  mean no encoder-discriminative signal. Replaces the ad-hoc
  "supervised beats floor by 0.10" gate.
- **Voita & Titov 2020, "Information-Theoretic Probing with MDL"** (EMNLP) —
  probing as compression (codelength of labels given representation). Trivial tasks
  have tiny codelength for any representation including random, so the
  real-vs-control gap collapses visibly. Capacity-robust; no hand-set ceiling.
- **Hewitt & Liang 2019, "Designing and Interpreting Probes with Control Tasks"** —
  selectivity = task accuracy − random-control-task accuracy. Already in the specs.
- **Pimentel et al. 2020, "Information-Theoretic Probing for Linguistic Structure"**
  (ACL) — the meaningful quantity is I(representation; property) relative to
  I(input; property), i.e. gain over raw input, not an absolute score.

### The coordinate-leakage mechanism

- **Jiang et al. 2023, "Self-Supervised Pre-Training With Masked Shape Prediction
  for 3D Scene Understanding"** (CVPR) — architecture must alleviate masked-shape
  leakage from point coordinates.
- **"Mitigating Positional Leakage in 3D Masked Autoencoders"** (MPL-MAE, 2026;
  treat as directional) — positional-embedding gradients dominate encoder-feature
  gradients; fixes are gated/sparse positional injection, recalibrated positional
  embeddings, omitted top-level skip connections.

### Connectivity / reachability probe design

- **Savinov et al. 2018, "Semi-Parametric Topological Memory for Navigation"**
  (ICLR) — positives within `l` steps, negatives separated by ≥ M·l steps (M=5
  margin). Without the margin, random negatives are trivially far apart and AUPRC
  saturates.
- **Eysenbach et al. 2019, "Search on the Replay Buffer"** (NeurIPS) — pair
  sampling must span the full geodesic range.
- **Emmons et al. 2020, "Sparse Graphical Memory"** — two-way / cycle-consistent
  reachability to suppress false-positive edges (= false-open).
- **Wu et al. 2017, "Sampling Matters in Deep Embedding Learning"**;
  **Robinson et al. 2021, "Contrastive Learning with Hard Negative Samples"**;
  **Schroff et al. 2015 (FaceNet)** — semi-hard negative mining; for a fixed eval,
  approximate by stratifying pairs by geodesic distance and forcing boundary pairs.
- **Arganda-Carreras et al. 2015 (VI / Adapted Rand); Stucki et al. 2023 (Betti
  matching)** — already in the pretraining spec; connected-component-structure
  metrics if pairwise AUPRC keeps saturating.

### The ceiling / supervised-reference problem

- **Papyan, Han, Donoho 2020, "Prevalence of Neural Collapse during the terminal
  phase of deep learning training"** (PNAS) — a plain classifier's penultimate
  features collapse to ≈ (#classes − 1) dims; cannot be used as an encoder ceiling.
- **Ishida et al. 2023, "Is the Performance of My Deep Network Too Good to Be
  True? A Direct Approach to Bayes Error Estimation"** (ICLR), code
  `github.com/takashiishida/irreducible`; **Renggli et al. 2021, "FeeBee"** —
  model-free irreducible-error / achievable-ceiling estimation (kNN, MST, or the
  direct estimator). No reference training, no collapse, no instability.
- **Lyle et al. 2022 (InFeR); Direct Singular Value Regularization; VICReg
  covariance term; Balestriero & LeCun 2025 (SIGReg / LeJEPA)** — anti-collapse
  regularizers if a trained reference is retained; SIGReg also targets training
  instability.

### Normalized-score / floor-ceiling gate design

- **Agelink van Rentergem et al. 2022, "Methods for Constructing Normalised
  Reference Scores"**; BIG-bench `(raw − low)/(high − low)` critiques — ceiling
  effects shrink discriminative variance; equal-weight aggregation lets
  mediocre-everywhere tie a spiky candidate. `pilot_score` (unweighted mean of five
  normalized components, three denominators degenerate) has exactly this problem.
- **Agarwal et al. 2021, rliable / "Statistical Precipice"** (in the specs);
  **Bowyer et al. 2025, "Statistical Uncertainty Quantification for Aggregate
  Performance Metrics in ML Benchmarks"** — per-metric IQM + performance profiles +
  confidence intervals instead of one normalized mean.

### Protocol amendment vs deviation

- Clinical-trial convention: a timestamped **amendment before confirmatory data**
  is not a **deviation**. P0C is calibration with no P1 candidate results, so a
  probe/anchor redesign is a clean amendment. The v1 P1 result has confirmatory
  data, so retiring it is "superseded by amendment; verdict void because the veto
  is shown unpassable by construction," referencing the P0C report hash.
- **Pineau et al. 2021, "Improving Reproducibility in ML Research"** — artifact set
  for the amendment: new spec commit, dataset id, run ids, code sha.

## 4. Alternative strategies

### A. `occupied_iou`

1. **Masked / held-out occupancy** — query only cells masked from the encoder
   input; PCA/random cannot reach 1.0. Closest to the current design; matches the
   MAE objective.
2. **Off-grid / sub-voxel queries** — continuous coordinates not aligned to input
   voxels; forces interpolation from features.
3. **Cross-channel** — build the embedding from `known_free` + `unknown` only,
   predict `occupied`; removes the identity path.
4. **Drop the metric** — rely on `boundary_f1` + `clearance_nmae`; occupancy IoU
   may simply be the wrong geometry probe.
5. **Change the score, not the task** — PVI / MDL gain over a coordinates-only
   baseline; triviality becomes visible and there is no hand-set ceiling.

### B. `reachability` / false-open

1. **Geodesic-distance-stratified pair sampling with a positive/negative margin**
   (SPTM M·l); report AUPRC per distance bin.
2. **Boundary-case negatives** — pairs whose connectivity flips under a 1–2 voxel
   obstacle perturbation.
3. **Two-way / cycle-consistent reachability** — count "connected" only if A→B and
   B→A agree; suppresses false-open.
4. **Connectivity-structure metric** (VI / Adapted Rand / Betti-0) instead of
   pairwise AUPRC.
5. **Empirical veto** — set the false-open gate relative to the calibrated baseline
   distribution ("beat the best non-neural baseline by X"), not an absolute 0.05.
6. **Threshold calibration** — pick the decision threshold on a held-out fold
   (Youden's J / cost-weighted); report threshold-free (AUPRC) and
   calibrated-threshold error rates separately.

### C. `geodesic`

1. **Fix normalization** — normalize by per-geometry maximum geodesic or a fixed
   physical scale; require the frequency-baseline NMAE to be well above the noise
   floor (for example > 0.15) before the metric is usable.
2. **Distance-stratified pair sampling** — bounded windows over-concentrate on
   short distances.
3. **Multi-horizon Spearman / ordinal consistency** (segments at, e.g., 20/50/100
   steps) instead of a single NMAE — planning-relevant and scale-robust.
4. **Log-distance or bucketed classification** — spreads a low-variance target.
5. **Defer to Stage 2** — bounded radius-8/16 windows may genuinely lack
   long-horizon geodesic structure.

### D. supervised reference / ceiling

- **Bayes-error estimators (recommended for classification metrics)** — model-free
  ceiling for `occupied_iou` and `reachability_auprc`; no training loop.
- **Multi-task reference** — one trunk, all five metric heads jointly, so there is
  no single simplex to collapse to; the trunk is then a legitimate strong-encoder
  ceiling.
- **Metric-achieved ceiling** — measure the ceiling as the task metric reached by a
  head-on-raw-input model, not by probing its penultimate features.
- **Regularized trained reference** — anti-collapse regularization (InFeR /
  DirectSVR / covariance) plus a fixed training recipe (warmup, cosine schedule,
  gradient clipping, weight EMA for evaluation, evaluation every 20–50 updates);
  accept only if `effective_rank_fraction > ~0.3`.
- **Over-capacity ensemble** — 3–5 references; ceiling = ensemble metric.

### v1 P1 retirement

1. **Formal supersede** — the spec commit records the v1 P1 verdict as void,
   citing this report's hash as proof the veto is unpassable by a random projection
   (0.148 ≫ 0.05); new run / dataset / preregistration ids; old artifacts marked
   `superseded_by`.
2. (weaker) mark "inconclusive under a defective probe" — leaves a muddier record.

## 5. Prioritization and assessment

### Priority P0 — cheap, unblocks everything

1. **Ceiling redesign → Bayes-error estimators for `occupied_iou` and
   `reachability_auprc`.** Removes the "train a reference that may collapse or
   diverge" failure mode. `occupied_iou` still needs step 2 because PCA reaches 1.0
   regardless of the reference.
2. **`occupied_iou` → masked / held-out occupancy queries + PVI/MDL scoring.**
   Structural and unavoidable; medium cost (query generation + metric swap).
3. **`reachability` → stratified + margin pair sampling, boundary negatives,
   empirical veto, threshold calibration.** Highest design cost of the three probes,
   highest leverage — the false-open veto is the P1 gate.

### Priority P1 — before P0D reopens

4. **`geodesic` → normalization fix + stratified sampling + multi-horizon
   Spearman**, or explicitly defer to Stage 2. Record the decision.
5. **Fix the reference training transient** (warmup, cosine, gradient clip, weight
   EMA, denser evaluation) if any trained reference remains in the loop.

### Priority P2 — record-keeping, in parallel

6. **Formal supersede of the v1 P1 result** with this report's hash; new spec
   commit and run / dataset ids.
7. **Retire the `pilot_score` unweighted mean** in favour of per-metric reporting
   plus IQM / performance profiles; gate on per-metric anchors only.
8. **Freeze `boundary_f1` and `clearance_nmae` as the calibration template**;
   document why they pass (real floor < ceiling headroom, monotone with the
   reference on top) as the acceptance bar for the reworked three metrics.

### Assessment and risks

- **Most likely post-fix outcome:** the two structural probes will show real
  headroom, but the pilot may still land near `no_viable_direction` — because
  2,000-update Tiny encoders genuinely do not produce strong probe discrimination
  at pilot horizon. That is the short-horizon pilot validity limit, not a probe
  bug; budget for it.
- **Neural collapse is a constraint, not a bug.** Any few-class supervised
  classifier's penultimate features collapse. If the protocol needs an *encoder*
  ceiling (not just a metric ceiling), it must be multi-task or
  self-supervised-with-labels.
- **Geodesic may not belong in the pilot at all.** Deferring it to Stage 2 (larger
  radius) is a legitimate scope call.
- **Do not over-fit the probes to pass the gate.** Verify the reworked tasks
  against Bayes-error / PVI baselines, not by tuning until
  `supervised − floor ≥ 0.10`.
- **Cost estimate:** P0 items ≈ 1–2 weeks including tests; re-running P0C/P0D is
  ~0.12 accelerator-hours. Main risk is scope creep into a full probe-suite
  redesign — hold the line at the three broken metrics.

## 6. What P0C did correctly

P0C measured real floors and ceilings (not toy fixtures), ran the ceiling gate,
and `blocked` instead of clamping a degenerate denominator, at 0.116 of a 2.0
accelerator-hour cap, with every artifact hashed (queries, four baseline model
states, supervised state, predictions, seven pool hashes) and
`calibration_used_for_candidate_ranking: false` recorded. It caught a
study-invalidating problem on the first run for about seven minutes of GPU time.
That is the intended behaviour of the gate.
