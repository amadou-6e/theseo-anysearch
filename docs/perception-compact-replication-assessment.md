# Fine729 replication assessment

Issue #401 / PR #402; spec `b77b6a20ffc7e88930b4e4110133a649afdb95c7`;
source `b3a28146fa2c2c8f890e9b0b9b531196032cb8a8`. Fresh run
`compact-replication-v1-run1` completed all12 fits and109 hashed artifacts.
Report `f7d65863a81c3c905d24649d78a4d2d55ce747ccc0c89d361feb796d4bfc7c10`.
[Full report](perception-encoder-local-geometry/compact-replication-report.json).

## Decision

The fixed selected fine729/LR.0003 recipe passes the preregistered experimental
packaging-readiness assessment in every development family for all three fresh
seeds. No failing seed was averaged away, no runner-up substituted, and no
threshold fitted on development. Proceed to frozen encoder/native-head packaging
checks, not promotion or a planning-readiness claim.

| Seed | Development IoU | Boundary F1 | Clearance NMAE | Recovery NMAE |
| ---: | ---: | ---: | ---: | ---: |
| 401 | .763331 | .791490 | .041957 | .041946 |
| 402 | .763984 | .794351 | .044363 | .044443 |
| 403 | .771223 | .794583 | .040736 | .040789 |

All-family/all-seed minima: IoU .631216 (thin sheets), F1 .705934 (random field).
All-family/all-seed maximum errors: clearance .059629, recovery .059573.
Thin-sheet IoU spans .631216-.658902 and F1 .771460-.794040.
Every seed/family native F1 exceeds both shuffled and fitted-null controls by
the required .10. Fitted-null pooled F1 is approximately .256; zeroed-code
occupancy and boundary scores are zero. Full per-family values are in the report.

## What this supports

Keep input33/central17 fixed. Preserve a spatially arranged1x9x9x9 latent,
flattened to729 float32 numbers, with probe-only normalization. Freeze backbone
and aggregation; use a separate trainable layout-aware head. The previous
common-head deficit remains a limitation, not something this replication retested.

This is conditional three-seed evidence on one fresh shared synthetic corpus and
one fixed pretrained backbone. The weakest F1 margin is only .005934. No broad
distributional confidence interval, arbitrary-head transfer, larger context,
topology, RLlib or navigation performance is established. Spatial references
have comparable pooled F1 (.7915-.7920), but thin-sheet IoU below .60 in this
corpus; they are not an oracle ceiling and their failures did not relax targets.

```mermaid
flowchart LR
  A[Selection-locked fine729 recipe] --> B[Fresh corpus and seeds401/402/403]
  B --> C[All family targets and code necessity pass]
  C --> D[Verify artifacts and export frozen encoder]
  D --> E[Separate trainable native head]
  E --> F[Experimental local-geometry use only]
```

## Validation

526 garden tests pass. Original-CUDA replay verified all109 artifacts, three
seed pipelines,30 predictions and eight backbone batches at unchanged tolerances.
Runtime accounting is
6,876.375 seconds (1.910 hours), including frozen900-second overhead and timed
preparation, below the four-hour tranche within the existing48-hour authorization.
No weights or corpora are committed; no PRs are merged.
