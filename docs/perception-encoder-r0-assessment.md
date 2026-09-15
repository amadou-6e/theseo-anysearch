# Completed v2r2 R0 assessment

Run: `voxel-encoder-pilot-v2r2-r0-1`, 2026-09-08.
Execution status: **completed**. Registered assessment: **defer**.
Recorded stop: **no_topology_identifiable**. Acceptance: pending protocol/result review.

The experiment ran to completion without changing its published settings. This is
an identifiability/control experiment, not a new P1 encoder-pretraining result.

## What ran

- Fresh native procedural volumes and the preregistered visible-conditioned
  empirical patch-completion distribution.
- 192 donor geometries and 720 preassigned candidate geometries. Fixed, label-blind
  eligibility/quota selection filled all 216 training and 144 audit slots.
- 64 completions per selected observation/query: 23,040 selected labels, including
  9,216 held-out audit labels. No final validation/test geometry was opened.
- Four controls fitted at 1,024 updates in each of two evaluation-domain paths:
  8,192 control updates. Both paths produced identically hashed fitted models.
- RTX 3060 Ti; 33.905 seconds total wall time. CUDA fitting occupied 8.039 seconds
  including overhead, an upper bound of 0.002233 accelerator-hours, below the
  registered 0.5-hour cap. No cap was exhausted.

## Why it deferred

All geometry quotas passed. The two shorter-span bins failed **class support**
in the held-out spatial-sampling configuration:

| Optimistic span | In-domain negatives / 1,536 | Held-out negatives / 1,536 | Required per domain |
|---|---:|---:|---:|
| 1-2 | 84 | 0 | At least 64 negatives and 64 positives |
| 3-5 | 79 | 0 | At least 64 negatives and 64 positives |
| 6+ | 117 | 154 | Met |

The `6+` bin had both classes, but all four registered interval checks failed:

| Check | Measured conservative bound | Requirement |
|---|---:|---:|
| Conditional entropy | Upper 0.915733 bits | < 0.90 |
| Visible skill over geometric prior | Lower -0.077859 bits/query | > 0.02 |
| Forbidden-metadata predictability | Upper 0.144125 bits/query | < 0.02 |
| Source-to-transfer skill drop | Upper 0.156454 bits/query | < 0.05 |

An interval failing a gate is not proof of the opposite hypothesis. In particular,
the held-out `6+` entropy point estimate was 0.447082 bits, not one bit of complete
randomness. Its uncertainty bound failed the registered requirement.

## What we learned

The source-domain visible predictor improved mean log loss over the geometric
summary prior by 0.123384, 0.116036 and 0.023719 bits/query across the three bins.
These are descriptive point differences, not passing overall audit decisions.
For held-out `6+`, it was 0.005864 bits/query worse than the prior.

The chosen distribution and transfer configuration do not supply a qualified
topology task for architecture selection. The missing negative labels in two
transfer bins are a direct observed failure, independent of model quality. Stride
two can remove one-voxel walls, but this audit does not causally isolate that
mechanism from patch selection or query placement.

The forbidden-metadata ablation did not meet the registered low-predictability
bound. It was not supplied to the visible model; this is not evidence of
encoder-input leakage. The audit also conditions on fixed donor banks and tests
an empirical patch prior, not the exact native-generator posterior.

No encoder objective or architecture was eliminated by these results. There was
no P1 training, and this small, single-recipe audit cannot establish universal
topology identifiability or its impossibility.

## Stop and review

The registered rule stops this execution before P0C, P0D and P1. No comparative
contract was frozen. Do not lower a threshold, replace a failed bin or switch to
AUPRC under these identities. #341 is now separately proposable, not started.

There is a **review-ordering deviation**. The earlier state-review snapshot asked
for review of numeric choices before data opening. The execution registration
was committed and pushed before opening, but external review of specs PR #28 was
still pending when the run started. That ordering expectation was missed before
launch. Publication preserves the pre-data choices; it does not retroactively
satisfy external approval. The immutable result remains evidence under the
published protocol, with acceptance or supersession for reviewers to decide.
Neither implementation nor specification PR has been merged.

## Provenance and reproduction

- Governing pilot spec: `0060366f43892bade5acf734cf2e189d8a666ad2`.
- Pre-data execution registration: `a0d03c434be72d6eab7d962e53af62a7dd57b11d`
  in `amadou-6e/specs`.
- Executable source: `f902b428c61b3a20c386fb3a91d28726fdd470d3`.
- Published protocol commit: `6c34230632920534b54bb4a625b784b7636d26dd`;
  the runtime HEAD differed from executable source only by the frozen contract.
- Protocol identity: `8230bd97ea14f14f8a8910af61cde7ad211a6c189c9129981d225b45c79a8fe3`.
- R0 report payload: `f14bacb88d29c1da009fb3ed02a6fa63648675706fef488661eb1c44dff9e71c`.

[Full report](../experiments/perception_encoder/results/v2r2_r0/r0-report.json),
[frozen protocol](../experiments/perception_encoder/results/v2r2_r0/audit-protocol.json),
[predictions](../experiments/perception_encoder/results/v2r2_r0/predictions.json),
[data manifest](../experiments/perception_encoder/results/v2r2_r0/data-manifest.json),
[control hashes](../experiments/perception_encoder/results/v2r2_r0/control-artifacts.json).

The assessment was reproduced exactly from saved predictions without regenerating
data or retraining. Regression tests verify artifact hashes, replay the decision,
and reject comparative registration after this deferral. All 354 Garden tests
passed after adding these checks:

```powershell
python -m pytest tests/test_garden/test_unit/test_v2r2_r0_result.py -q
```
