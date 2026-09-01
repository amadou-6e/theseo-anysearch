# Perception Encoder Experiment Work Plan

Status: active
Integration branch: `exp/perception-encoder`
Specifications:

- [Pilot plan](https://github.com/amadou-6e/specs/blob/f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d/projects/theseo-anysearch/python/perception-encoder-pilots.md)
- [Pretraining protocol](https://github.com/amadou-6e/specs/blob/f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d/projects/theseo-anysearch/python/perception-encoder-pretraining.md)
- [Architecture review](https://github.com/amadou-6e/specs/blob/f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d/projects/theseo-anysearch/python/perception-encoders.md)

The pilot preregistration is pinned to specs commit
`f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d`. A later methodology revision does not
silently change an active run. Adopting one requires a recorded deviation, a new frozen
preregistration artifact, and new dataset/run identities before affected results open.

## Objective

Implement and execute the preregistered perception-encoder pilots without coupling the
work to RLlib or allowing exploratory changes onto `develop`. Each row below becomes a
GitHub issue and `exp/<issue-number>` branch. Infrastructure tasks deliver tested code;
execution tasks deliver locked configurations, compact result reports, artifact hashes,
and a decision record.

## Work packages

| ID | Issue and branch | Work package | Primary ownership | Depends on |
|---|---|---|---|---|
| F0 | [#272](https://github.com/amadou-6e/theseo-anysearch/issues/272), `exp/272` | Pilot contracts, resolved configuration, run manifest, frozen-preregistration, and decision-record schemas | `garden/pilots/`, narrow additions to `garden/data_config.py` | none |
| F1 | [#273](https://github.com/amadou-6e/theseo-anysearch/issues/273), `exp/273` | Named geometry-disjoint pools, immutable split/query hashes, reproducible fresh draws, micro-scenes, and exact occupancy/ESDF/topology targets | `garden/collect.py`, `garden/dataset.py`, new `garden/targets.py` and `garden/splits.py` | F0 (#272) |
| F2a | [#274](https://github.com/amadou-6e/theseo-anysearch/issues/274), `exp/274` | Frozen probes, cross-fitting, selectivity, and embedding-necessity controls | `garden/evaluation/probes.py`, `garden/evaluation/controls.py` | F0 (#272) |
| F2b | [#284](https://github.com/amadou-6e/theseo-anysearch/issues/284), `exp/284` | Topology/collapse diagnostics, paired geometry bootstrap, and calibrated learning-curve reporting | `garden/evaluation/metrics.py`, `garden/evaluation/statistics.py`, `garden/evaluation/curves.py` | F0 (#272) |
| F3 | [#275](https://github.com/amadou-6e/theseo-anysearch/issues/275), `exp/275` | Encoder output contract and Tiny dense residual, tri-planar, shared-pyramid, and optional sparse backbones | `garden/models/backbones.py`, `garden/models/outputs.py` | F0 (#272) |
| F4 | [#276](https://github.com/amadou-6e/theseo-anysearch/issues/276), `exp/276` | Update-based training runtime, T0-T3 objective wrappers, dense mask-aware fallback, optional sparse path, isolation gates, and resource accounting | `garden/trainer.py`, new `garden/models/objectives.py` and `garden/masking.py` | F0 (#272) |
| F5 | [#277](https://github.com/amadou-6e/theseo-anysearch/issues/277), `exp/277` | P0 contract gate, populated frozen-preregistration artifact, and reproducible pilot command | `garden/pilots/runner.py`, micro-scene integration tests, CLI wiring | F1, F2a, F2b, F3, F4 (#273-#276, #284) |
| E1 | [#278](https://github.com/amadou-6e/theseo-anysearch/issues/278), `exp/278` | Execute P1-P2 objective and training-mechanics pilots | locked configs and reports under `experiments/perception_encoder/` | F5 (#277) |
| E2 | [#279](https://github.com/amadou-6e/theseo-anysearch/issues/279), `exp/279` | Execute P3 random-weight architecture feasibility profiling | locked configs and reports under `experiments/perception_encoder/` | F3 (#275) |
| E3 | [#280](https://github.com/amadou-6e/theseo-anysearch/issues/280), `exp/280` | Execute P4 architecture signal and P4D observation-density checks | locked configs and reports; no new unscoped framework code | E1-E2 (#278-#279) |
| E4 | [#281](https://github.com/amadou-6e/theseo-anysearch/issues/281), `exp/281` | Execute P5 field-of-view micro-ablation | locked configs and reports | E3 (#280) |
| E5 | [#282](https://github.com/amadou-6e/theseo-anysearch/issues/282), `exp/282` | Execute P6 objective-by-architecture interaction check | locked configs and reports | E1, E3-E4 (#278, #280-#281) |
| E6 | [#283](https://github.com/amadou-6e/theseo-anysearch/issues/283), `exp/283` | Execute P7 fresh-seed confirmation and P8 radius-128 viability smoke | locked configs, final pilot decision, artifact hashes | E5 (#282) |
| PR | [#285](https://github.com/amadou-6e/theseo-anysearch/issues/285), deferred `develop/285` | Selectively promote accepted foundation/implementation commits and compact reports | promotion inventory and existing owned modules only | E6 (#283) |

If an execution task exposes a reusable implementation defect, open a new focused issue
instead of expanding that experiment branch. Failed and inconclusive runs still produce
their preregistered decision record.

## Parallel schedule

```text
F0
 |-- F1 data and targets ----------|
 |-- F2a probes and controls ------|
 |-- F2b metrics and statistics ---|--> F5 P0 gate --> E1 P1/P2 --|
 |-- F3 encoder backbones --> E2 P3 ------------------------------|--> E3
 `-- F4 training and objectives ---|                                  |

E3 P4/P4D --> E4 P5 --> E5 P6 --> E6 P7/P8 --> PR selective promotion
```

After F0 merges, F1, F2a, F2b, F3, and F4 can run in parallel because their primary file
ownership is separate. E2 starts as soon as F3 merges and overlaps the other foundation
work: random-weight profiling needs neither data/objectives nor the P0 gate. E1 waits
for F5. E3 waits for both E1 and E2; E3-E6 are otherwise decision-dependent and remain
ordered.

F0 defines `experiments/perception_encoder/preregistration.yaml`; F5 must freeze its
fully populated instance before P1. It pins the exact specs repository and commit,
per-pilot and total accelerator-hour caps, seed assignments, veto thresholds,
pilot-score floor/ceiling anchors, and geometry-ID membership and hashes for every named
development pool and fresh draw. Placeholders or post-result mutation block P1.

## Integration gates

Every infrastructure PR must provide unit tests and one deterministic CPU smoke path.
CUDA/sparse dependencies must be optional and report an explicit skipped or unavailable
state. The dense mask-aware fallback owns the deterministic CPU hidden-value,
hidden-Jacobian, and mask-only-shortcut fixtures; sparse isolation is additionally
verified under CUDA. F5 may report the sparse CUDA path skipped on CPU, but it cannot
skip the dense correctness gates. F5 cannot pass until mask-leakage, frozen-state,
target-oracle, output-shape, statistical-oracle, preregistration, and decision-record
fixtures all pass.

Every execution PR must include:

- the issue, integration-base SHA, code SHA, dataset/query hashes, and resolved config;
- completed, failed, and skipped trial counts against the preregistered cap;
- per-geometry metrics and bootstrap inputs by artifact reference;
- peak memory, latency, updates, observations, valid voxels, and accelerator-hours;
- a machine-readable decision that follows the pinned rule from the recorded numbers,
  or a logged deviation citing a new specification revision and run identity;
- the exact veto/rule behind every rejection and every validity limit that could
  plausibly invert the decision;
- a complete record for `no_viable_direction`, `blocked`, failed, and inconclusive
  outcomes as well as winner/tie outcomes; and
- an explicit `promote`, `retain`, or `reject` recommendation.

Execution issues own configs, runs, and reports only. They may fix a defect local to the
experiment, but must open a focused infrastructure issue for reusable framework work
rather than silently expanding scope.

No execution result can change a threshold after its evaluation pool is opened. Changes
to preregistered behavior require a specification revision and a new dataset/run version.

## Promotion boundary

Issue #285 tracks promotion but its `develop/285` branch is not created until E6 closes.
Cherry-pick only the foundation/implementation commits required by the accepted
direction plus compact, useful reports. Do not promote losing backbones, abandoned
objective code, model weights, raw datasets, or the entire integration history. A
`no_viable_direction` or `blocked` outcome promotes foundation work and useful reports
only. The promotion PR to `develop` reruns all garden unit tests, relevant CLI tests, and
deterministic P0 smoke tests on the current development line.

The pilot program selects a direction and does not publish a model checkpoint. Interim
`experimental` garden publication remains a separate post-Stage-2 action governed by
the pinned pretraining specification; #285 must not relabel a pilot artifact as a
published or validated encoder.
