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

## Remaining before an evidential R0 run

1. Implement and validate the full-3D conditional-completion generator and its
   query sampler. Preserve observed voxels and group all sibling completions.
   Do not substitute the retained `(1, 15, 15)` exploratory fixture.
2. Implement the control-fitting producer, geometry-disjoint preprocessing,
   artifact provenance and held-out-generator execution. Keep forbidden
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

CUDA checked on 2026-09-08: PyTorch `2.13.0+cu126`, NVIDIA GeForce RTX 3060 Ti.
No v2r2 audit observations have been generated or opened by this increment.
No audit protocol or comparative preregistration has been frozen, no R0 result
has been emitted, and no P1 training has started. v2r1 remains terminal.

Disposition: `retain` (infrastructure). Keep #340 open and the PR in draft until
the execution items above are complete. Nothing is merged without review.
