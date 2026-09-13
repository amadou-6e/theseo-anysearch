# Obstacle-rich compiled-world waypoint preflight

Issue [#409](https://github.com/amadou-6e/theseo-anysearch/issues/409) tests
whether the empty-grid waypoint pretraining recipe archived in PR #217 can be
used on a compiled world with genuine detours. This is a feasibility fixture,
not a training result or a change to the preregistered perception-encoder pilot
at specs commit `a94227bc4ee484287a026f89ec6cd47d5ca16d26`.

Run from the repository root:

```powershell
python usage/experiments/train/large_world_obstacle_pilot/preflight.py --samples-per-stage 2
```

The deterministic source in `preflight.py` is a 128 x 96 x 64 world with three
two-voxel-thick partitions, staggered 16 x 12 doorways, and three interior
blocks. The compiler writes the pack and report only under ignored
`runtime/obstacle-waypoint-pilot/`. The first pack identity is
`ff8c1e946e90525ef9c9270757d37316c63f19cea2c0d9ae47717ad7d94165ab`;
its manifest records 43,755 occupied voxels.

The preflight samples two deterministic 96-action curriculum routes at each of
the 11 configured segment-distance stages (1, 3, ..., 19, 20). It checks every
endpoint against compiled occupancy, searches every segment with the existing
lazy A* planner, checks PR #217's 128-step episode budget, and checks the exact
`shortest_actions` that the current `continue_route` imitation collector uses.

| Check | Routes |
| --- | ---: |
| Sampled | 22 |
| A* feasible and within 128 steps | 20 |
| Direct empty-grid actions collision-free | 16 |
| Random route contains an occupied waypoint | 2 |
| A* feasible but direct actions cross geometry | 4 |

For seed `415001`, the direct actions cross geometry but A* finds a 114-step
route. Executing that A* plan in the actual compiled-world environment reached
all waypoints without collision in 114 steps. The raw per-seed report is
`runtime/obstacle-waypoint-pilot/preflight.json` and is intentionally not
committed.

This demonstrates why copying PR #217's YAML directly is unsafe. Its imitation
collector's `_route_action_plan` takes the `continue_route` branch and calls
`shortest_actions`; it does **not** invoke the configured `replanning_astar`
provider in that branch. Behavior cloning still runs, but its teacher actions
are not obstacle-aware. The existing geometry-pool A* reset-feasibility option
does not validate these separately sampled compiled-world waypoint routes.

Before a training comparison, generate or reject routes using occupancy and
bounded A* feasibility, and make the imitation collector execute the planned
paths. Freeze separate scratch and imitation run configurations, seeds, pack
identity, teacher budget, and evaluation routes. Do not treat this preflight as
evidence that a policy has learned the obstacle task.
