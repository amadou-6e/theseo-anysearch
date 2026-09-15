# Original-budget six-gate rerun

This is issue [#409](https://github.com/amadou-6e/theseo-anysearch/issues/409)
on the `exp/409` branch, governed by the
[pilot spec](https://github.com/amadou-6e/specs/blob/a94227bc4ee484287a026f89ec6cd47d5ca16d26/projects/theseo-anysearch/python/perception-encoder-pilots.md)
at `a94227bc4ee484287a026f89ec6cd47d5ca16d26`. The source budget and
evaluation cadence come from the archived as-run configuration in
[PR #217](https://github.com/amadou-6e/theseo-anysearch/pull/217) at
`e81ca9765709d5e34e2f46ab35726eaedc130e68`. That historical run planned
400 PPO iterations and was interrupted after iteration 340. This new run
uses the full original **planned** cap, not the separate 2/2 pipeline smoke.

`experiment.yaml` is frozen with SHA-256
`d90bea74d3429fb7e970f1c1914b2194a7e7c6525fafef6752d16777db8f08cf`.
It requests 128 A*-teacher demonstrations, 20 behavior-cloning
epochs (with the original five-epoch early-stopping patience), 400 PPO
iterations, 4096 transitions per PPO batch, three rollout workers with four
environments each, checkpoints every 50 iterations, ten regular evaluation
episodes, and three episodes per curriculum stage every five iterations.
The original sparse `segment_countdown_goal` reward is rebuilt from the same
Rust reward function and parameters; its tiny-environment test verifies a
10-point optimal segment completion.

The necessary world adaptation is explicit: the 32-cubed empty world and
96-action sampled segment curriculum are replaced by the compiled
4096 x 2048 x 512 six-gate world (identity
`ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05`)
and 12 exact stages from 2 through 4096 actions. Episode limit is 4608,
not 128. The 12 fixed anchors use seeded final-waypoint variants so the
stratified collector can obtain 128 distinct successful demonstrations.
`astar` plans once per waypoint segment rather than the original
`replanning_astar` at every action; this preserves an obstacle-aware teacher
without thousands of redundant searches per long route. Dataset reuse is
disabled to give this world a fresh dataset. This is a large-world adaptation
of the original training budget, not a controlled reproduction of the
historical empty-grid outcome.

The generated dataset, extension build, checkpoints, logs, and model weights
remain ignored under `runtime/` or `.anysearch/`. Run from the `exp409`
worktree root using its source checkout:

```powershell
C:\CodeWorkspace\theseo-anysearch\.venv\Scripts\python.exe -c "from theseo_anysearch.cli.main import app; app()" run usage/experiments/train/large_world_obstacle_pilot/original_budget/experiment.yaml
```

Do not confuse this config or its run identity with the completed
`gate-stratified-smoke/fb696335` feasibility run, which used 2/2 rather than
the original budget.
