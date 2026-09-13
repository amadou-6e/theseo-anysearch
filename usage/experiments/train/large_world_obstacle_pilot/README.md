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
4,294,967,296 logical cells, 266 box sources, and 7,014,727 occupied
voxels. Six full YZ-section walls at X=512, 1152,
1792, 2432, 3072, and 3712 divide the long X axis. Each one-voxel-thick
wall has a centered square portal, shrinking in travel order through 32 x 32,
16 x 16, 8 x 8, 4 x 4, 2 x 2, and finally **1 x 1 voxel** at
`(3712, 1024, 256)`. The walls span the entire Y and Z ranges, so the final
portal is the only crossing at that X plane. Sixteen translated local obstacle
structures remain at widely separated positions for bounded waypoint-route
preflight. Two long, thin obstacles also cross the central space. The compiler
writes the pack and report only under ignored `runtime/obstacle-waypoint-pilot/`.
The pack identity is
`bd9b1982b02bc43b0cbbb5c06ab8925463ea96197b99d77f0116f00831c308f4`.
This fixed-route pilot writes an empty candidate index: enumerating every
surface voxel on the full-section walls would exhaust memory, and no
candidate-index-based spawn/goal provider is used here. Occupancy, rendering,
and A* planning still use the complete compiled geometry.

The preflight samples two deterministic 96-action curriculum routes at each of
the 11 configured segment-distance stages (1, 3, ..., 19, 20) in four widely
separated regions. Starts are `(144,176,128)`, `(1424,176,384)`,
`(2704,1904,128)`, and `(3856,1904,384)`. It checks every
endpoint against compiled occupancy, searches every segment with the existing
lazy A* planner, checks PR #217's 128-step episode budget, and checks the exact
`shortest_actions` that the current `continue_route` imitation collector uses.

| Check | Routes |
| --- | ---: |
| Sampled | 88 |
| A* feasible | 74 |
| A* feasible and within 128 steps | 73 |
| Direct empty-grid actions collision-free | 55 |
| Random route contains an occupied waypoint | 14 |
| A* feasible but direct actions cross geometry | 19 |
| A* feasible but above the episode budget | 1 |

For seed `415001`, the direct actions cross geometry but A* finds a 114-step
route. Executing that A* plan in the actual compiled-world environment reached
all waypoints without collision in 114 steps. The raw per-seed report is
`runtime/obstacle-waypoint-pilot/preflight.json` and is intentionally not
committed. The report also records six short A* wall crossings and confirms
that the final crossing uses `(3712, 1024, 256)`.

## Preview the obstacles

From the repository root, generate four replayer views plus static global and
local close-up images without enumerating the compiled voxel volume:

```powershell
python -m usage.experiments.train.large_world_obstacle_pilot.preview --images
```

Open `runtime/obstacle-waypoint-pilot/previews/global_obstacles.png` to see the
full-width wall positions and 16 separated local regions. Open
`portal_progression.png` to compare the six openings in equal-scale YZ
cutaways, and `local_obstacles.png` for one local region's partitions and
blocks. The latter layout is translated to all 16 regions.

The four `region_*.json` and six `portal_*.json` files can be opened together
with `voxel-replay` for interactive inspection of the actual compiled pack.
Use `[` and `]` to switch views; the global overview is enabled by default,
and the regional view is centered near each route start or portal. These are
geometry-only previews, not recorded training episodes.

This is a *large-extent, local-route* test. The 96-action episodes sample far
apart starting regions, but no single episode traverses thousands of voxels.
It validates regional loading and obstacle-aware local routing at widely
separated coordinates; it does not validate long-distance navigation across
the entire world or train a policy to traverse the portal sequence. The six
short A* crossing probes establish portal connectivity only. During an earlier
boundary-adjacent attempt, an A* query at
the top Z coordinate raised a native out-of-bounds error. The retained fixture
keeps routes away from that edge; boundary behavior needs its own follow-up.

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
