# P1-P2 objective and training-mechanics report

P1 status: `completed`

P2 status: `blocked`

The full preregistered P1 matrix completed on the reference RTX 3060 Ti: 16 of
16 trained cells completed, with zero runtime failures and zero OOMs. No bundle
passed every hard gate, so the locked P1 decision is `no_viable_direction` and
P2 records `P1_retained_no_bundles` without opening its training matrix.

## P1 result

| Bundle | Mean pilot score | Effective-rank fraction | False-open rate | Rejection gates |
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

## Consequence

P2 correctly emitted a blocked decision instead of selecting training mechanics
from an ineligible bundle. Although P3 independently retained four architecture
families on feasibility, P4 must not train them because the frozen sequence
requires a valid P2 training bundle. P4 therefore terminates as
`no_viable_direction`; starting it would be an unregistered protocol deviation.

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
