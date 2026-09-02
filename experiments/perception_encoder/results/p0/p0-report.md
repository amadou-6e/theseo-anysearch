# P0 contract and overfit smoke

P0 completed on an NVIDIA GeForce RTX 3060 Ti with PyTorch 2.13.0+cu126. The
execution used code commit `4e15fd96885bbe3c39e7575171b7df2d058c3a48` and the
specification commit `f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d`.

## Decision

P0 selects no winner. T0, T1, T2, and T3 are retained for P1 because every bundle
completed 500 updates with finite losses, changed encoder state, and reduced its
fixed micro-set loss by at least 90%.

| Bundle | Initial loss | Final loss | Reduction |
|---|---:|---:|---:|
| T0 occupancy | 1.051811 | 0.000138 | 99.987% |
| T1 masked occupancy | 0.682011 | 0.000353 | 99.948% |
| T2 truncated ESDF | 0.303199 | 0.000508 | 99.832% |
| T3 EMA latent target | 0.314434 | 0.002657 | 99.155% |

All target-oracle, metric-behavior, paired-bootstrap, learning-curve, frozen-state,
mask-isolation, output-contract, and architecture-gradient gates passed. The legacy
current encoder and the dense residual, tri-planar, shared-pyramid, and dense
mask-aware candidates each completed 20 finite forward/backward batches and changed
trainable state. The mask intervention error, hidden-input Jacobian, and mask-only
shortcut advantage were all zero.

## Resources and identity

- Accelerator time: 0.013952 hours.
- Preregistration contract SHA-256: `26c6bd55a8442f33463efceff0ccd2de71c0c68287677b22c4132862dc6f285b`.
- Preregistration file SHA-256: `f9e6394140746618f4ec2934ee90aae728f12a1f8b9fb90a1e14bdf457e5ff97`.
- Report payload SHA-256: `ac2318d292a5249e2eeb60cde63d94e0062f2bde009647b2f56b42e679d70eb5`.
- JSON file SHA-256: `0243b38d3bf4c283b8ec6824d0dac422200a3e4a3cd5fc879260f145c3c75647`.

## Validity

This is direction-finding evidence only, not encoder certification. The imported
geometry identity is a labeled synthetic mesh-import fixture, not external
out-of-family evidence. P1 must fit normalization floors and supervised ceilings on
the frozen corpus rather than treating P0's analytic calibration fixtures as measured
corpus baselines.
