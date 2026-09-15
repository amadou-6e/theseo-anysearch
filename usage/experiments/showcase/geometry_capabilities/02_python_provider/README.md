# Python geometry provider

`geometry.py`'s `procedural_walls` places `wall_count` full-height walls at
random x positions, each pierced by a one-voxel gap at a random y, seeded
entirely from `context.seed`. `env.geometry.provider` selects it by name.

```powershell
anysearch geometry sample usage/experiments/showcase/geometry_capabilities/02_python_provider/experiment.yaml --count 20 --seed 42 --output samples.json
```

Verified: all 20 samples come back `geometry: valid`, `task: feasible`, each
with a different `metadata.walls` layout (deterministic per seed -- re-run
with the same `--seed` and `samples.json` is byte-identical). Every sample
also carries `geometry_identity`, `task_feasibility.difficulty`, and the
full `proposal` the provider returned, so acceptance/rejection and
suitability are inspectable per sample without starting Ray.

## Replay one accepted sample

```powershell
python usage/experiments/showcase/geometry_capabilities/02_python_provider/replay_accepted_sample.py --seed 42
anysearch replay file runtime/geometry_capabilities/02_python_provider/replay/trajectories/heuristic_astar.json
```

Re-resolves seed 42's geometry, walks it with the A* heuristic oracle
(verified: 58 steps, success), and writes a replayer-compatible trajectory
via the same `write_heuristic_trajectory` a training run uses.
