# Perception audit repairs (#350)

Governing predecessor spec: `d9b260a5c677277d47776c7677fe07c2a7ff4f34`.
Companion methodology amendment and visual state map: specs#25.
Reviewed amendment snapshot: `cbcd34a8991a2eb8cae88f100d2f081992dfd437`
in `amadou-6e/specs` (pending review; not a frozen execution protocol).
The diagram is maintained in the specs repository at
`projects/theseo-anysearch/python/perception-encoder-state-review.md`.

- Path span now follows a deterministic shortest path with lexicographic next-cell
  tie-breaking and includes unknown endpoints. Previous screen values were based
  on the union of shortest paths; they remain archived and non-evidential.
- P0D and terminal P1 verify canonical payload content against the pinned hash,
  rather than trusting the report's self-declared hash.
- `geometry_held_out_posteriors` provides empirical reference predictions with
  per-fold preprocessing and complete geometry exclusion. #340 must assign all
  counterfactual siblings to the same fold and use visible inputs for the gate.
  Historical v2r1 estimators and results are preserved.
- The work-plan diagnosis identifies the actual coordinates-only winning floor.

Disposition: retain. This is prerequisite infrastructure, not encoder quality
evidence. #340 must implement the v2r2 contracts and audit runner, pin the merged
spec amendment and actual execution code, and freeze numeric audit parameters
before opening audit data. No P1 training is authorized by these helper fixes.

Validation: 292 Garden tests passed. Archived P0C and P0D payloads also passed
the strengthened transition checks. Specification validation passed with one
unrelated existing large Draw.io file warning.
