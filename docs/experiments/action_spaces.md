# Action-space experiments

This document records controlled action-space comparisons, including both
successful and unsuccessful runs. Runtime artifacts remain under `runtime/` and
are not committed.

## Twenty-voxel overfit comparison

Date: 2026-07-28

### Question

Can PPO overfit a fixed, axis-aligned 20-voxel navigation task, and how does the
action representation affect convergence?

### Shared configuration

| Setting | Value |
|---|---|
| Grid | 32 x 32 x 32 |
| Start | `(4, 4, 4)` |
| Goal | `(24, 4, 4)` |
| L1 and L2 waypoint distance | 20 voxels |
| Maximum episode length | 50 steps |
| Observation | `radial`, `ray_max_len: 10` |
| Trail mode | disabled |
| PPO iterations | 50 maximum |
| Training EnvRunners | 8 |
| Evaluation episodes | 20 deterministic episodes per iteration |
| Early stop | 20 goal finishes in one evaluation batch |
| Seed | 42 |
| Step cost | -0.5 |
| Collision cost | -0.02 |
| Goal reward | 2.0 |
| Distance shaping | 0.25, progress mode |

The table above records the resolved run configuration. Future runs use four
evaluation EnvRunners. The `discrete_26` result below used
eight because it was launched before that default was reduced; `vector_3` used
four.

### Results

| Action mode | Run ID | Evaluation workers | Outcome | First successful iteration | Goal finishes | Successful steps | Wall time |
|---|---|---:|---|---:|---:|---:|---:|
| `discrete_26` | `6d0ba2e1` | 8 | Solved | 20 | 20/20 | 20 | 190.6 s |
| `vector_3` | `d29bb1f7` | 4 | Not solved | - | 0/20 at iteration 50 | - | 229.5 s |

The successful `discrete_26` policy used the optimal 20-step axis-aligned path.
Its run stopped at iteration 20.

### `vector_3` failure analysis

The factorized `MultiDiscrete([3, 3, 3])` policy learned to approach the goal but
never entered it:

| Iteration | Mean minimum goal distance | Mean final goal distance |
|---:|---:|---:|
| 1 | 14.53 | 29.27 |
| 20 | 1.41 | 3.74 |
| 30 | 4.12 | 4.24 |
| 40 | 2.24 | 2.24 |
| 50 | 1.00 | 1.00 |

At iteration 50 it reached `(23, 4, 4)`, one voxel before the goal, then
alternated between `(23, 4, 4)` and `(23, 4, 5)`. The recorded canonical actions
were `(0, 0, -1)` and `(0, 0, +1)` instead of the required final `(1, 0, 0)`.

This suggests that independently selecting three axis categories can make exact
coordination harder than selecting one categorical movement from the 26 valid
offsets. It is evidence from one seeded run, not yet a statistical conclusion.
A multi-seed comparison is required before treating the action representation as
the sole cause.

### Evaluation-performance observation

The earlier `discrete_26` run with serial evaluation reached the 50-iteration
ceiling without a success and took 834.9 seconds. After moving evaluation to
RLlib evaluation workers, the successful run took 190.6 seconds, a 77.2 percent
wall-time reduction. The learning outcomes are not directly attributable to
evaluation parallelism because distributed PPO execution is not perfectly
deterministic. The reliable conclusion is the evaluation-throughput improvement.

### Current conclusion

- `discrete_26` is the strongest representation in this fixed 20-voxel test.
- `vector_3` learned useful goal-directed behavior but failed at the final action.
- Four dedicated evaluation workers are the default for subsequent comparisons.
- Repeat each action space across multiple seeds before making a general ranking.
