# Reference refinement: a supported training direction

Issue #364. Disposition: retain. Completed screen and fresh three-seed assessment.
No promotion, encoder updates, integration merges or historical verdict changes.

## Main result

The selected recipe was dynamic independent masks/queries, 192 training geometries,
AdamW learning rate 0.003, weight decay 0.01, and selection-best checkpointing.
Reference architecture, batch size eight and 2048-update budget stayed fixed.

| Seed | Baseline F1 | Refined F1 | Frozen probe F1 | Baseline loss | Refined loss | Probe loss |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.4897 | 0.7628 | 0.6275 | 0.2440 | 0.1243 | 0.1794 |
| 1 | 0.4879 | 0.7805 | 0.6182 | 0.2366 | 0.1186 | 0.1780 |
| 2 | 0.4697 | 0.7468 | 0.6376 | 0.2523 | 0.1339 | 0.1788 |

F1 thresholds and probe capacities were selected only on selection data. All
refined models chose update 512; all baseline models chose update 128. Both still
trained for the full budget. Mean refined F1 is 0.7634 versus baseline 0.4824.

Paired assessment log-loss gains, positive favoring refinement:

- Versus baseline: 0.1187, geometry-cluster 95% CI [0.1023, 0.1318].
- Versus frozen probe: 0.0531, geometry-cluster 95% CI [0.0296, 0.0740].

Outcome: `reference_exceeds_probe`. Report keys inherited from the checkpoint
comparison call chosen recipe `best` and baseline `final`; both are checkpoint-
selected in this campaign, not final-update models.

## Screen, not confirmation evidence

| Recipe | Best selection loss | Best update |
| --- | --- | --- |
| Fixed masks, 48 geometries | 0.2550 | 128 |
| Dynamic masks, 48 geometries | 0.1570 | 1024 |
| Dynamic masks, 192 geometries | 0.1473 | 1024 |
| Above with lower learning rate | 0.1486 | 2048 |
| Above with higher weight decay | 0.1470 | 1024 |
| Above with learning rate 0.003 | 0.1292 | 1024 |

This one-seed screen suggests fresh masking is the largest tested contribution,
but does not establish independent population-level effects for every component.
The fresh three-seed comparison confirms the selected bundle against baseline.
Selection was persisted before any confirmation fitting or assessment.

## Direction for subsequent experiments

Prioritize training-data and objective design before increasing architecture size
or voxel radius. Keep dynamic corruption, broader geometry coverage and early
selection monitoring as the working reference recipe. The fixed-mask reference's
poor result was not a useful ceiling on what the visible input supports.

Next scoped encoder study should compare the existing frozen encoder against an
encoder pretrained using broader geometry and fresh corruption, with architecture
held fixed. First equalize probe exposure at 192 training geometries for both
encoders; include the old 48-geometry probe as a control. This is needed because
the current reference uses 192 geometries but the probe retained its original 48.
Current results do not prove that the encoder discarded information, or isolate
architecture from pretraining history. Freeze these settings on fresh identities
before fitting; do not tune on this campaign's assessment data.

Use per-family surface scores, distance-task retention and paired uncertainty to
decide whether encoder retraining merits scaling. Do not relax gates or start a
large architecture tournament merely because a pooled reference score improved.

## Remaining limits

Oblique-sheet refined F1 is 0.6925/0.7226/0.6613, still below 0.70 in two seeds.
The pooled scores clear 0.70, but this is not a universal family-level pass or
encoder certification. The assessment consists of fresh instances of the same
four synthetic development families. No topology, planning, wider-radius,
unseen-family or real-world claim follows. No distance tasks were retrained here.

## Reproducibility

- Frozen spec `44b089b58db3b9af45820cb358033c6a4acbc99f` in specs.
- Executable source `7f3727c`, stacked on exp/362 with user authorization.
- Recorded duration 256.16 seconds; RTX 3060 Ti / PyTorch 2.13.0+cu126.
- Six screen fits, six confirmation reference fits, nine probe fits; D0 passed.
- [Frozen manifest](perception-encoder-local-geometry/refinement-preregistration.json).
- [Full report](perception-encoder-local-geometry/refinement-report.json).
- Payload hash `0baf857fccb0bc18ecb8c4aa917a2028d20d4625941a69f0de1460be94801043`.
- Artifacts remain untracked in `runtime/reference-refinement-v1-run1`; hashes in
  report. No weights or generated corpora committed.
