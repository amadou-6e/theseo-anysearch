# Perception Encoder Experiment Work Plan

Status: proposed
Integration branch: `exp/perception-encoder`
Specifications:

- `specs/projects/theseo-anysearch/python/perception-encoder-pilots.md`
- `specs/projects/theseo-anysearch/python/perception-encoder-pretraining.md`
- `specs/projects/theseo-anysearch/python/perception-encoders.md`

## Objective

Implement and execute the preregistered perception-encoder pilots without coupling the
work to RLlib or allowing exploratory changes onto `develop`. Each row below becomes a
GitHub issue and `exp/<issue-number>` branch. Infrastructure tasks deliver tested code;
execution tasks deliver locked configurations, compact result reports, artifact hashes,
and a decision record.

## Work packages

| ID | Work package | Primary ownership | Depends on |
|---|---|---|---|
| F0 | Pilot contracts, resolved configuration, run manifest, and decision-record schema | `garden/pilots/`, narrow additions to `garden/data_config.py` | none |
| F1 | Geometry-disjoint corpus, immutable split/query hashes, micro-scenes, and exact occupancy/ESDF/topology targets | `garden/collect.py`, `garden/dataset.py`, new `garden/targets.py` and `garden/splits.py` | F0 |
| F2 | Frozen probes, controls, topology/collapse metrics, paired geometry bootstrap, and calibrated learning-curve reporting | new `garden/evaluation/` package | F0 |
| F3 | Encoder output contract and Tiny dense residual, tri-planar, shared-pyramid, and optional sparse backbones | `garden/models/backbones.py`, `garden/models/outputs.py` | F0 |
| F4 | Update-based training runtime, T0-T3 objective wrappers, sparse/mask-aware path, mask isolation, and resource accounting | `garden/trainer.py`, new `garden/models/objectives.py` and `garden/masking.py` | F0 |
| F5 | P0 contract gate and reproducible pilot command integrating F1-F4 | `garden/pilots/runner.py`, micro-scene integration tests, CLI wiring | F1, F2, F3, F4 |
| E1 | Execute P1-P2 objective and training-mechanics pilots | locked configs and reports under `experiments/perception_encoder/` | F5 |
| E2 | Execute P3 architecture feasibility profiling | locked configs and reports under `experiments/perception_encoder/` | F5 |
| E3 | Execute P4 architecture signal and P4D observation-density checks | locked configs and reports; no new unscoped framework code | E1, E2 |
| E4 | Execute P5 field-of-view micro-ablation | locked configs and reports | E3 |
| E5 | Execute P6 objective-by-architecture interaction check | locked configs and reports | E1, E3, E4 |
| E6 | Execute P7 fresh-seed confirmation and P8 radius-128 viability smoke | locked configs, final pilot decision, artifact hashes | E5 |

If an execution task exposes a reusable implementation defect, open a new focused issue
instead of expanding that experiment branch. Failed and inconclusive runs still produce
their preregistered decision record.

## Parallel schedule

```text
F0
 |-- F1 data and targets ---------|
 |-- F2 evaluation and statistics|--> F5 P0 gate
 |-- F3 encoder backbones --------|
 `-- F4 training and objectives --|

F5 --> E1 P1/P2 ----|
  `--> E2 P3 -------|--> E3 P4/P4D --> E4 P5 --> E5 P6 --> E6 P7/P8
```

After F0 merges, F1-F4 can run in parallel because their primary file ownership is
separate. E1 and E2 can also run in parallel after P0 passes: objective screening does
not need P3 profiling, and random-weight feasibility profiling does not need the P2
winner. E3-E6 are decision-dependent and must remain ordered.

## Integration gates

Every infrastructure PR must provide unit tests and one deterministic CPU smoke path.
CUDA/sparse dependencies must be optional and report an explicit skipped or unavailable
state. F5 cannot pass until mask-leakage, frozen-state, target-oracle, output-shape, and
decision-record fixtures all pass.

Every execution PR must include:

- the issue, integration-base SHA, code SHA, dataset/query hashes, and resolved config;
- completed, failed, and skipped trial counts against the preregistered cap;
- per-geometry metrics and bootstrap inputs by artifact reference;
- peak memory, latency, updates, observations, valid voxels, and accelerator-hours;
- the machine-readable decision record and a concise Markdown interpretation; and
- an explicit `promote`, `retain`, or `reject` recommendation.

No execution result can change a threshold after its evaluation pool is opened. Changes
to preregistered behavior require a specification revision and a new dataset/run version.

## Promotion boundary

After E6, open one promotion issue. Cherry-pick only the foundation/implementation
commits required by the accepted direction plus compact, useful reports. Do not promote
losing backbones, abandoned objective code, model weights, raw datasets, or the entire
integration history. The promotion PR to `develop` reruns all garden unit tests, relevant
CLI tests, and deterministic P0 smoke tests on the current development line.
