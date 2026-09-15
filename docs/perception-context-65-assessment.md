# Frozen 33 to 65 context assessment

Issue #374. Disposition: retain. No integration merge or promotion.
Spec: amadou-6e/specs@2f9fa46f3c26f96b9f0e1c05f9fa101a09923ad3,
`projects/theseo-anysearch/python/perception-context-65.md`.
Source f848db6, authorized stack on unmerged #372. Run voxel-context-65-v1-run1.

## Result

All four paired noninferiority checks pass. None has a positive lower95% gain
bound. Shared frozen weights remain usable at65, but this study finds no clear
benefit from increasing33 context to65 for central17 local-geometry queries.

| Task | 33 seed scores | 65 seed scores | Improvement95% CI |
| --- | --- | --- | --- |
| Occupied IoU | .6570/.6801/.6786 | .6625/.6782/.6828 | [-.00124,.00678] |
| Boundary F1 | .7440/.7411/.7299 | .7446/.7284/.7472 | [-.00551,.00883] |
| Clearance NMAE | .03075/.03397/.03342 | .03025/.03386/.03402 | [-.000527,.000482] |
| Recovery NMAE | .03052/.03438/.03393 | .03011/.03443/.03435 | [-.000571,.000523] |

Improvement reverses sign for errors. Whole-parent bootstrap within six strata,
paired seeds and queries,2000 draws; noninferiority margins .03/.02. These are
conditional intervals over this small corpus and three fixed encoder seeds, not
universal architecture certification. Per-family results remain in the report.

## Resources and direction

Batch4 passed first; no fallback. Total allocated CUDA peak107.70 MiB at33 and
610.98 MiB at65; incremental peaks72.31 and559.20 MiB. Local forward times
13.25-15.60 versus86.70-86.75 ms/batch4. Timing is not an isolated production
benchmark; CUDA allocation excludes external allocations. Corpus stayed on CPU.
Complete run including manifest preparation:84.797 seconds. No encoder updates.

Prefer33 context for this task pending wider-output validation:65 costs more
without demonstrated gain. The next bounded test is shared33-context tiled
inference against dense65 on wider spatial queries, measuring quality and cost;
do not assume tiling preserves features under spatial normalization or saves
compute after overlap. It requires its own fresh protocol/data identity.

## Evidence

Registration: `perception-encoder-local-geometry/context65-preregistration.json`.
Report: `perception-encoder-local-geometry/context65-report.json`.
Payload SHA256: `36fef437a4d62fd505e27c39cba7b58c393bc2ca1d387e426d96f0f9ce12704c`.
Full garden suite414 passed before the additional completed-report replay test.
Source checkpoints verified before execution and unchanged after each scale.
24 untracked probe/prediction artifacts carry SHA256 hashes; weights/corpus stay
outside Git. Separate scale-adapted heads were fitted. No full-volume, topology,
planning, radius512 or promotion claim follows.
