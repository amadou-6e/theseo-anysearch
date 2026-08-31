# All-stage waypoint policy-pretraining run

This record describes the no-trail PPO run that first solved the full configured
96-action waypoint route consistently after balanced, all-stage imitation
pretraining.

## Provenance

| Field | Value |
| --- | --- |
| Run ID | `56d364e3` |
| Experiment | `ppo-waypoint-ed7db-00008-all-stage-policy-pretraining` |
| Tuning source | `ed7db_00008` |
| Code revision used | `821424d` on the historical `fix/203` worktree |
| Started | 2026-08-20 16:15 CEST |
| Planned iterations | 400 |
| Last completed iteration | 340 |
| Completion status | Interrupted; the stale `run.json` still says `RUNNING` |

The generated runtime directory remains local at
`.worktrees/fix-203/experiments/train/004_all_stage_policy/runtime/train/ppo-waypoint-ed7db-00008-all-stage-policy-pretraining/56d364e3`.
Runtime data is intentionally not part of this archive.

## Configuration

- PPO with a `voxel_encoder` model and hidden sizes `[512, 256, 128]`.
- One-third GPU allocation, three environment runners, and four environments per
  runner.
- Empty 32-cubed grid, box observation radius 1, and discrete-18 actions.
- Trail mode disabled.
- Fixed route length of 96 actions with a 128-step episode budget.
- Segment-distance curriculum: 1, 3, 5, ..., 19, 20.
- Balanced imitation dataset of 128 accepted routes across all 11 meaningful
  curriculum stages.
- `replanning_astar` generation, 20 behavior-cloning epochs, and policy/encoder
  handoff. The value head was not initialized from pretraining.
- PPO parameters were copied from tuning trial `ed7db_00008`.

`experiment.as-run.yaml` preserves the exact historical configuration. The
current `experiment.yaml` migrates only the imitation block from the removed
`teacher` schema to the equivalent generation and sampling provider schema.
The archived file matches the source YAML byte-for-byte with SHA-256
`9c3e2f4ebfa2b033bc8bc85a356ef7cbf7507a5413187e48efdc392f82d05f8d`.

## Results

The curriculum reached the final meaningful stage, stage 10, at iteration 50.
At the scheduled curriculum evaluation on iteration 335:

- Stage 10 passed 3 of 3 routes.
- Every evaluated stage reported 3 of 3 successes.
- The retention gate passed.

The regular evaluation at iteration 340 reported:

| Metric | Result |
| --- | --- |
| Successful episodes | 10 of 10 |
| Success rate | 1.0 |
| Mean episode length | 96 steps |
| Mean return | 960.0 |
| Collision rate | 0.0 |
| Route completion fraction | 1.0 |
| Route efficiency | 1.0 |

All ten evaluation episodes completed the 96-action route in exactly 96 steps.

## Curriculum-counter caveat

The historical run recorded stage 67 at iteration 340. That number does not
represent a harder task than stage 10. The configured maximum segment distance
was 20, so stage 10 was the last distinct difficulty. The old controller kept
incrementing the stage number and reevaluating duplicate maximum-distance stages.

The first erroneous transition was stage 10 to stage 11 at iteration 55. By the
iteration-335 retention evaluation, the artifact contained 67 nominal stages
numbered 0 through 66, all above stage 10 repeating the capped difficulty.

This defect was corrected by issue #209 and PR #213. Current code stops
progression at the final meaningful stage. PR #211 also formalized balanced,
unique all-stage demonstration collection, and PR #212 stabilized heterogeneous
evaluation route suites. Therefore, a current rerun should reproduce the task and
policy settings but should remain at terminal stage 10.

## Reproduction

Run the current configuration from the repository root:

```powershell
anysearch experiment run usage/experiments/train/ppo_waypoint_ed7db_00008_all_stage_policy_pretraining/experiment.yaml
```

The run requires a CUDA-capable GPU. Dataset, checkpoint, trajectory, log,
TensorBoard, Ray, and MLflow artifacts remain under ignored `runtime/` paths.
