# P1-P2 objective and training-mechanics report

P1 status: `completed`

P2 status: `blocked`

The full preregistered P1 matrix completed on the reference RTX 3060 Ti: 16 of
16 trained cells completed, with zero runtime failures and zero OOMs. No bundle
passed every hard gate, so the locked P1 decision is `no_viable_direction` and
P2 records `P1_retained_no_bundles` without opening its training matrix.

## P1 result

| Bundle | Diagnostic mean pilot score | Effective-rank fraction | False-open rate | Rejection gates |
|---|---:|---:|---:|---|
| T0 occupancy | 0.4550 | 0.0188-0.0369 | 0.1536-0.2157 | rank, false-open |
| T1 masked occupancy | 0.3714 | 0.1493-0.1876 | 0.1821-0.2386 | rank, false-open |
| T2 ESDF | 0.5464 | 0.0333-0.0656 | 0.2127-0.2434 | rank, false-open |
| T3 latent target | -1.5091 | 0.0297-0.0614 | 0.1681-0.2774 | rank, embedding necessity, false-open, component improvement |

All values above summarize both learning rates and both seeds. T1 materially
improved dimensional rank relative to the other bundles, but its best value,
0.1876, remained below the frozen 0.25 minimum. T2 had the highest mean pilot
score but failed both safety gates, demonstrating why point score alone is not
a valid selection criterion. T3 also failed embedding necessity and the
three-component improvement requirement.

## Protocol interpretation

The P0 analytic normalization anchors are not measured baselines on this corpus,
so the aggregate pilot scores in the table are published for audit only and are
suppressed as selection or cross-bundle ranking evidence. The P1 `completed` status
means that the locked execution matrix finished; it does not establish that the
anchor-dependent comparison was well-conditioned.

The eligibility result does not rely on those anchors. Every trained cell failed
both `effective_rank_fraction >= 0.25` and `false_open_rate <= 0.05`, whose observed
values are raw metrics. Correcting only the score anchors would therefore leave the
recorded P1 eligibility result unchanged. A repair that changes the target, probe,
or corpus requires a new run identity and may change the raw metrics; that choice is
tracked in issue #322 rather than being made after seeing these results.

This is a screen at the frozen 2,000-update pilot horizon, not evidence that any
objective is incapable at the full Stage 2 budget. Short horizons can favor
fast-starting objectives and miss delayed representation gains.

## T3 training triage

T3 passed the P0 micro-set overfit gate with a 99.155% loss reduction, so its P1
result is not evidence that the objective implementation cannot optimize. All four
P1 runs reached their lowest sampled loss at update 1,000 or 1,500 and then rebounded
by the final checkpoint:

| Learning rate | Seed | Lowest sampled loss (update) | Update-2,000 loss | Rebound |
|---:|---:|---:|---:|---:|
| 0.0001 | 0 | 0.014491 (1,000) | 0.064041 | 4.42x |
| 0.0001 | 1 | 0.014189 (1,500) | 0.023604 | 1.66x |
| 0.0003 | 0 | 0.004498 (1,500) | 0.020047 | 4.46x |
| 0.0003 | 1 | 0.010693 (1,000) | 0.025740 | 2.41x |

Combined with the poor effective rank and embedding-necessity result, this warrants
triage of the EMA schedule, target normalization, late optimization stability, and
shortcut or collapse behavior. It does not support rejecting latent-target
pretraining as a general approach.

## Consequence

P2 correctly emitted a blocked decision instead of selecting training mechanics
from an ineligible bundle. Although P3 independently retained four architecture
families on feasibility, P4 must not train them because the frozen sequence
requires a valid P2 training bundle. P4 remains blocked with zero trials; starting
it would be an unregistered protocol deviation. Issue #322 owns the deliberate
choice between calibration repair and rerun, methodology revision, or accepting
the null result and stopping the program.

## Reproducibility

- Specification commit: `f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d`
- Integration base: `0b8476f90a5da45b060bd641daedb8f5468d0189`
- Execution code: `8c916ddc7e47c3962df0cc56d626db8eb0f541c9`
- Resolved config SHA-256: `c6c3e5f9d798a8dea953978a9e0322d81efb5aa51bd9574bf6b7588584e9251b`
- Preregistration SHA-256: `703aa4911f2e36f815fcee2a0c10f7fa9880a89c4d558c599e8b274e4cacc47b`
- P1 report SHA-256: `A81A5CB8C539DF90618EDCAD46D91877ED85CAE6D1251D304EA7F04EAF902D7C`
- P2 report SHA-256: `88C1E41C83F62D6126EECAC108E3760A0B81F410381D672B4064DC2B19CE95F1`
- Recorded P1 execution time: 2.522 accelerator-hours
- Runtime: PyTorch 2.13.0+cu126, CUDA 12.6, FP32, batch size 2

The recorded execution time covers the successful resumed matrix. The earlier
interrupted partial attempt consumed additional unreported wall time but did not
produce or alter a candidate artifact; total observed P1 time remained below the
four-hour cap. Raw per-trial JSON remains under `runtime/perception_encoder/p1_p2`
and is intentionally excluded from version control. The committed machine reports
contain hashes and references for every raw artifact.
