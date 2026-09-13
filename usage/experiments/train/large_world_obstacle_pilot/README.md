# Obstacle-rich compiled-world waypoint preflight

Issue [#409](https://github.com/amadou-6e/theseo-anysearch/issues/409) tests
whether the empty-grid waypoint pretraining recipe archived in PR #217 can be
used on a compiled world with genuine detours. This is a feasibility fixture,
not a training result or a change to the preregistered perception-encoder pilot
at specs commit `a94227bc4ee484287a026f89ec6cd47d5ca16d26`.

Run from the repository root:

```powershell
python -m usage.experiments.train.large_world_obstacle_pilot.preflight --samples-per-stage 2
```

The deterministic source in `preflight.py` is a 4096 x 2048 x 512 world:
4,294,967,296 logical cells, 24 box sources, and 6,290,091 occupied voxels.
**Only six portal walls** remain. They span the full YZ section at X=512,
1152, 1792, 2432, 3072, and 3712, dividing the long X axis. Each one-voxel-thick
wall has a centered square portal, shrinking in travel order through 32 x 32,
16 x 16, 8 x 8, 4 x 4, 2 x 2, and finally **1 x 1 voxel** at
`(3712, 1024, 256)`. The walls span the entire Y and Z ranges, so the final
portal is the only crossing at that X plane. Sixteen translated local obstacle
structures and the central cross have been removed; no geometry unrelated to
the gates remains. The compiler writes the pack and report only under ignored
`runtime/obstacle-waypoint-pilot/`.
The pack identity is
`ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05`.
This pilot writes an empty candidate index: enumerating every
surface voxel on the full-section walls would exhaust memory, and no
candidate-index-based spawn/goal provider is used here. Occupancy, rendering,
and A* planning still use the complete compiled geometry.

## Twelve-stage gate curriculum

`curriculum.py` defines exactly 12 fixed, success-advanced routes. Stage `i`
(zero-based) has `2^(i+1)` optimal actions: **2, 4, 8, 16, 32, 64, 128, 256,
512, 1024, 2048, 4096**. Each starts at `(1, 1024, 256)` and follows the
portal centers along X. The final route crosses all six gates. The native
occupancy query rejects X=4096, so it ends its X traverse at 4095 and adds
two Y actions after the final gate to reach exactly 4096 actions. The episode
limit is 4608 actions, leaving 512 actions of slack; the curriculum cap itself
is 4096. The preflight checks exact action lengths, source occupancy, and
collision-free direct teacher actions. An integration test executes all 4096
actions in the compiled-world environment and reaches the final goal.

The trainer now accepts explicit `routes` under the existing
`completion_mode: continue_route`. `gate_curriculum_settings()` returns the waypoint-curriculum
config block. Evaluation may repeat a frozen route with separate environment
seeds rather than trying to sample distinct routes that do not exist. This is
configuration and validity evidence, not a training result.

The separate stochastic feasibility probe still samples two deterministic
96-action routes at each of 11 segment-distance settings (1, 3, ..., 19, 20)
beside each of the six gates. Starts are four voxels before each wall at the
common portal center `(Y=1024, Z=256)`. It checks every
endpoint against compiled occupancy, searches every segment with the existing
lazy A* planner, checks PR #217's 128-step episode budget, and checks the exact
`shortest_actions` that the current `continue_route` imitation collector uses.

| Check | Routes |
| --- | ---: |
| Sampled | 132 |
| A* feasible | 100 |
| A* feasible and within 128 steps | 84 |
| Direct empty-grid actions collision-free | 59 |
| Random route contains an occupied waypoint | 32 |
| A* feasible but direct actions cross geometry | 41 |
| A* feasible but above the episode budget | 16 |

The raw per-seed report is `runtime/obstacle-waypoint-pilot/preflight.json` and
is intentionally not committed. It also records six short A* wall crossings
and confirms that the final crossing uses `(3712, 1024, 256)`. Seed `412001`
completed an A*-planned 112-step route in the live compiled-world environment
without collision.

## Preview the obstacles

From the repository root, generate six replayer views plus static global and
aperture-progression images without enumerating the compiled voxel volume:

```powershell
python -m usage.experiments.train.large_world_obstacle_pilot.preview --images
```

Open `runtime/obstacle-waypoint-pilot/previews/global_obstacles.png` to see the
six full-width wall positions. Open `portal_progression.png` to compare the
openings in equal-scale YZ cutaways.

The six `portal_*.json` files can be opened together
with `voxel-replay` for interactive inspection of the actual compiled pack.
Use `[` and `]` to switch views; the global overview is enabled by default,
and the regional view is centered near each portal. These are
geometry-only previews, not recorded training episodes.

The stochastic 96-action probe samples near each gate, but does not validate
long-distance policy navigation. The new 4096-action route validates the
teacher and environment along the complete sequence, not that a policy can
learn it. The six short A* crossing probes establish portal connectivity.
During an earlier
boundary-adjacent attempt, an A* query at
the top Z coordinate raised a native out-of-bounds error. The retained fixture
keeps routes away from that edge; boundary behavior needs its own follow-up.

This demonstrates why copying PR #217's YAML directly is unsafe. Its imitation
collector's `_route_action_plan` takes the `continue_route` branch and calls
`shortest_actions`; it does **not** invoke the configured `replanning_astar`
provider in that branch. The new fixed routes are intentionally aligned with
the gates, and preflight proves those direct actions collision-free. Random
gate-adjacent routes are not generally safe for this collector. The existing
geometry-pool A* reset-feasibility option does not validate separately sampled
compiled-world waypoint routes.

The fixed gate-axis routes are validated for the existing direct-action teacher.
If a comparison uses random routes instead, generate or reject them using
occupancy and bounded A* feasibility, and make the imitation collector execute
the planned paths. Before either training comparison, freeze separate scratch
and imitation run configurations, seeds, pack identity, teacher budget, and
evaluation routes. Do not treat this preflight as evidence that a policy has
learned the obstacle task.

## Bounded imitation-then-PPO smoke

`experiment.yaml` freezes a small validation run on this pack: one demonstration
per fixed stage (12 total), at most 12 collection attempts, two behavior-cloning
epochs, and two PPO iterations. Each distinct fixed route is collected once;
the validation episode is held out by episode. The run uses the radius-1
voxel-encoder PPO settings from PR #217, except that its unavailable custom
reward is replaced with the built-in progress reward and its rollout batch is
reduced to 1024 for the bounded smoke. Its 4608-step episode limit applies to
every stage. Curriculum retention evaluation is scheduled after the smoke's
two iterations, so this run tests pretraining and PPO execution, **not**
12-stage policy mastery or curriculum advancement. The frozen world identity
is `ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05`.
Run from this worktree root with:

```powershell
python -c "from theseo_anysearch.cli.main import app; app()" run usage/experiments/train/large_world_obstacle_pilot/experiment.yaml
```

The fixture and run are governed only as feasibility work by the pinned
[perception-encoder pilot spec](https://github.com/amadou-6e/specs/blob/a94227bc4ee484287a026f89ec6cd47d5ca16d26/projects/theseo-anysearch/python/perception-encoder-pilots.md).
Runtime dataset, checkpoints, and logs remain ignored; record their IDs and
hashes in the issue/PR after execution. A longer comparison requires a
separately frozen compute budget and scratch control.
