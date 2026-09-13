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
This fixed-route pilot writes an empty candidate index: enumerating every
surface voxel on the full-section walls would exhaust memory, and no
candidate-index-based spawn/goal provider is used here. Occupancy, rendering,
and A* planning still use the complete compiled geometry.

The preflight samples two deterministic 96-action curriculum routes at each of
the 11 configured segment-distance stages (1, 3, ..., 19, 20) beside each of
the six gates. Starts are four voxels before each wall at the common portal
center `(Y=1024, Z=256)`. It checks every
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

This is a *large-extent, gate-local-route* test. The 96-action episodes sample
near each gate, but no single episode traverses thousands of voxels. It
validates gate-adjacent route feasibility, not long-distance navigation across
the entire world or policy traversal of the whole portal sequence. The six
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
