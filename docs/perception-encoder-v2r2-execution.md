# v2r2 execution progress (#340)

Branch: `exp/340`, based on experimental integration commit
`2844775ffda682c51d414a3b8f80e01b8d958975` (review repairs, #351).

Governing specification:
[perception-encoder-pilots.md at 0060366](https://github.com/amadou-6e/specs/blob/0060366f43892bade5acf734cf2e189d8a666ad2/projects/theseo-anysearch/python/perception-encoder-pilots.md).
This supersedes the issue's original reference to the unmerged specs#23 proposal.
The foundation pin remains `f03509f926d217cf8a0c8c34cd0da38020a5034c`;
the eventual audit also needs its actual executable commit SHA.

## Increment 1: assessment primitives

`garden/pilots/v2r2_audit.py` implements:

- Explicit required numerical assessment settings without acceptance defaults.
- Boolean 3D counterfactual checks: visible cells cannot change between
  completions; observed input cannot contain hidden occupancy.
- One prediction per fixed observation/query across all its hidden completions.
- Geometry-disjoint fitting manifests, sibling-group consistency, and held-out
  generator configuration checks.
- Conditional-entropy estimates with an exact-binomial probability envelope.
  A finite constant-label sample does not imply zero population entropy.
- Prior, coordinate-only, visible-context and forbidden-feature log losses in
  bits, with the specified probability clipping.
- Equal-geometry, then equal-context aggregation. Bootstrap samples entire
  geometries within caller-specified family/density strata, not completion rows.
- Separate class/count support checks in each domain and occlusion stratum.
- Conservative interval-based entropy, visible-skill, forbidden-predictability
  and generator-transfer assessment. Missing support or unresolved evidence
  produces `defer`; there is no best-stratum fallback.
- Content-addressed assessment output explicitly marked
  `r0_assessment_not_execution_report` and `authorizes_comparative_run: false`.

The estimator assumes IID conditional completions. It is not valid to supply
correlated MCMC samples and call the binomial envelope exact. Population
cluster-bootstrap coverage remains approximate; resampling preserves declared
family/density counts. Fitting manifests are checked for consistency, but the
future producer must verify that fitted artifacts actually match those manifests.

All numerical settings in unit tests are synthetic test inputs, not proposed or
frozen acceptance thresholds. Their `feasible` cases are not research evidence.

## Increment 2: control fitting and CUDA development check

`garden/pilots/v2r2_controls.py` now produces held-out assessment records from
validated 3D counterfactual volumes. Completed volumes supply six-connected
reachability labels only. The visible model receives flattened observed occupancy,
the unknown mask and endpoint coordinates; forbidden metadata has a separate
ablation input. Observation identity is checked independently of query identity,
so changing endpoints cannot move the same observation into another fold.

The current explicit recipe fits a linear geometric-summary prior, a linear
coordinate control, a visible-input MLP and a forbidden-feature MLP. It uses
training-only, geometry-weighted standardization and full-batch AdamW with a
fixed final checkpoint. The soft-label loss equals the completion-averaged BCE,
without duplicating voxel inputs for each completion. Neither evaluation labels
nor evaluation features affect fitting, preprocessing or checkpoint selection.
These recipes are implemented options, not yet frozen scientific choices or
the calibrated R2 control ladder.

Fitting returns geometry/configuration manifests, training-input hashes, model
and preprocessing hashes, parameter counts and resource accounting. CUDA requests
fail explicitly when unavailable; there is no silent CPU fallback. Unit tests
cover input isolation, held-out preprocessing, sibling leakage, RNG restoration,
numerical input validation, explicit device failure and wall-budget enforcement.

Development-only CUDA report:
`experiments/perception_encoder/results/v2r2_development/control-smoke-cuda-2.json`.

- Executable source: `d6d4500bbf7913a3ca92fe176b35e4fc5ec0249e`, clean at execution.
- Payload SHA-256: `9e3f9a3e2235a66810eac15dda22c165b849c06595d0d41799749f2b4ed00707`.
- Toy `17 x 17 x 17` inputs, 32 training geometries, 12 evaluation geometries
  per domain, 16 completions per context.
- Four controls, 128 updates each, in each of two domains: 1,024 control updates.
- RTX 3060 Ti; fitting wall times 1.844 s and 0.405 s, including device/fitting
  overhead. These small-fixture timings are not an R0 throughput prediction.
- `non_evidential: true`, `r0_executed: false`, zero candidate-training updates.
  The toy completion rule and nominal configuration labels test execution, not
  conditional identifiability or generator-transfer quality.

Validation after the observation-level grouping repair: 335 Garden tests passed
(28 v2r2 assessment/control tests); compilation and whitespace checks passed.

## Completed R0

R0 was executed and assessed on 2026-09-08 after source, specification registration
and the resolved protocol were committed and pushed. It returned `defer`, with
recorded stop `no_topology_identifiable`. See the
[full assessment and review-ordering deviation](perception-encoder-r0-assessment.md).

All 216 training and 144 audit geometry quotas were filled. Two held-out bins had
zero negative labels; the `6+` bin failed all four conservative interval checks.
No threshold was adjusted, no comparative contract was frozen, and no P0C/P0D/P1
was started. The assessment reproduces exactly from the saved predictions.

The following was the implementation sequence; its conditional later stages are
now stopped by R0, not a pending queue of authorized experiments:

1. Implement and validate the full-3D conditional-completion generator and its
   query sampler. Preserve observed voxels and group all sibling completions.
   Do not substitute the retained `(1, 15, 15)` exploratory fixture.
2. Wire the implemented control-fitting producer to that generator and the
   complete frozen fitting recipes. Validate real generator folds, query/bin
   assignment and fitting-artifact provenance end to end. Keep forbidden
   metadata out of visible-context inputs; oracle labels are not features.
3. Complete both v2r2 contract schemas. Freeze the audit/calibration protocol
   first, including fresh identity manifests, generator definitions, numerical
   counts/thresholds, R2 recipes, R3 veto calibration rule, R4 deterministic
   selection rule, component retention and separate resource caps.
4. Commit the executable implementation, then pin its SHA and the specification
   above before opening audit data. Run R0 with resource accounting and a hashed
   report; only a qualifying R0 can open P0C.
5. After qualifying P0C, freeze the comparative contract. P0D then gates P1.
   Deferral must preserve the audit and terminate under the topology rule; it
   must not be converted to an architecture verdict or silently start #341.

## Execution state

CUDA execution on 2026-09-08: PyTorch `2.13.0+cu126`, NVIDIA GeForce RTX 3060 Ti.
The registered R0 completed in 33.905 wall seconds, with 8,192 control updates
and zero candidate updates. Its audit protocol is frozen; its result is stored
under `experiments/perception_encoder/results/v2r2_r0/`. v2r1 remains terminal.

Disposition: `retain` (infrastructure and negative audit evidence, pending review).
Keep #340 open until its integration result is reviewed and accepted. External
protocol review was pending at execution; the deviation is explicit in the
assessment. Nothing is merged without review.
