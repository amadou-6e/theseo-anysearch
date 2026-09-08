# Perception audit repairs (#350)

Governing predecessor spec: `d9b260a5c677277d47776c7677fe07c2a7ff4f34`.
Companion methodology amendment and visual state map: specs#25.
Reviewed amendment snapshot: `d31fd6f03ab32f13b51863eb38e94adbcd971c77`
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

Follow-up review: accept NumPy integral neighbour counts (reject booleans and
floats), use one backward BFS per span, and share canonical hashing through
`pilots.io.payload_sha256` across the P0C/P0D/P1 evidence chain.

Validation: 307 Garden tests passed. Archived P0C, P0D, and P1 payloads passed
the strengthened transition checks. Specification validation passed with one
unrelated existing large Draw.io file warning.
