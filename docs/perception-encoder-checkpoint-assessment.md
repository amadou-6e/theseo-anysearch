# Checkpoint-selection diagnostic assessment

Issue #362. Disposition: retain. Completed development experiment; no promotion.

## Result

Outcome: `reference_improves_below_probe`. All three seeds selected update 256,
the earliest checkpoint in the preregistered candidate set. Each trajectory still
trained for the complete 2048-update budget. Selection never used assessment data.

| Seed | Final log loss | Selected log loss | Probe log loss | Final F1 | Selected F1 | Probe F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6070 | 0.3508 | 0.1887 | 0.3676 | 0.3829 | 0.6006 |
| 1 | 0.6012 | 0.3351 | 0.1920 | 0.3736 | 0.3990 | 0.5847 |
| 2 | 0.5715 | 0.3351 | 0.1918 | 0.3865 | 0.4017 | 0.5699 |

F1 uses each model's selection-calibrated threshold. Probe capacity was selected
by selection log loss from the unchanged linear/32/128 ladder. Lower loss is better.

Paired geometry-cluster-bootstrap intervals, averaging across seeds:

- Final loss minus selected loss: mean 0.2529, 95% CI [0.2339, 0.2725].
- Probe loss minus selected loss: mean -0.1495, 95% CI [-0.1639, -0.1353].

Checkpoint selection therefore helps probability quality, but does not close the
surface-recovery gap. Its F1 benefit is modest. The frozen encoder's features remain
more useful under the tested readout than this supervised-from-scratch reference.
This does not establish a general encoder ceiling or prove any missing target
information is unobservable.

## Direction

Refine the reference before using it to judge encoder limitations. All selected
checkpoints were the earliest allowed candidate, so the best checkpoint could be
earlier still; this experiment did not evaluate that claim. A next bounded reference
study can test earlier checkpoint cadence and training regularization/data coverage
as separately controlled interventions, not retrospectively change this run.
Do not infer that a larger encoder is needed from these results.

The comparison is paired within this fresh campaign, not against old D0-D2 metrics.
It has 48 training /24 selection /48 assessment geometries from the same four
development generator families, not a new unseen-family confirmation. No topology,
planning, radius-scaling or real-world claims follow.

## Audit

- Recorded elapsed time: 71.39 seconds, RTX 3060 Ti, PyTorch 2.13.0+cu126.
- D0 passed. Three matched reference trajectories and nine probe fits completed.
- Zero encoder updates; checkpoint file/state hashes verified and states unchanged.
- Frozen protocol: `df765d0d4eed2f74a1d6e85f358e435e126d4b08` in specs.
- Executable source: `d7b8e7a`, stacked on exp/360 under user authorization.
- [Registration](perception-encoder-local-geometry/checkpoint-preregistration.json).
- [Full report](perception-encoder-local-geometry/checkpoint-report.json).
- Payload SHA256: `72f39596cef5d24bf12ab2469c9950c1e7bab6411ea93436e97be58f9cee7f42`.
- Fifteen weight/prediction artifacts remain untracked under
  `runtime/checkpoint-diagnostic-v1-run1`; hashes recorded in the report.

No integration branch merges or historical verdict changes.
