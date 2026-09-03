# Geometry capabilities

A short progression from small procedural worlds to a large compiled one,
every step inspectable before training:

```powershell
anysearch geometry inspect <experiment.yaml>
anysearch geometry validate <experiment.yaml> --json
anysearch geometry sample <experiment.yaml> --count 20 --seed 42 --output samples.json
```

| Step | Demonstrates |
|---|---|
| [`01_composed_sources/`](01_composed_sources/) | Typed source composition (`env.geometry.sources`: an STL room unioned with a box obstacle) and feasibility diagnostics. |
| [`02_python_provider/`](02_python_provider/) | A `geometry.py` provider generating seeded, deterministic obstacle layouts; `sample --count 20` for acceptance/rejection and suitability across the distribution. |
| [`03_compiled_world/`](03_compiled_world/) | Content-addressed compiled-world artifacts and bounded (non-materializing) validation, with a valid and a deliberately blocked route side by side. |
| [`04_native_provider/`](04_native_provider/) | A native Rust geometry provider mirroring a Python reference: ABI discovery, identical canonical output, deterministic enforcement, malformed-provider diagnostics. |

## The primary live demo

The most coherent single story runs through step 2 then step 3 -- generate
diverse geometry, prove the task is solvable, preserve its identity, then run
the same shape of check at large-world scale:

```powershell
# Generate 20 deterministic obstacle layouts, with feasibility and difficulty
# reporting for each, without starting Ray:
anysearch geometry sample usage/experiments/showcase/geometry_capabilities/02_python_provider/experiment.yaml --count 20 --seed 42 --output samples.json

# Pick one accepted scene (seed 42) and replay the agent actually solving it:
python usage/experiments/showcase/geometry_capabilities/02_python_provider/replay_accepted_sample.py --seed 42
anysearch replay file runtime/geometry_capabilities/02_python_provider/replay/trajectories/heuristic_astar.json

# Repeat the same validate-then-replay shape against a compiled, non-cubic,
# 128x96x64 world -- and see the CLI actually tell a solvable route apart
# from a deliberately blocked one:
python usage/experiments/showcase/geometry_capabilities/03_compiled_world/prepare_world.py
anysearch geometry validate usage/experiments/showcase/geometry_capabilities/03_compiled_world/experiment_valid.yaml --json
anysearch geometry validate usage/experiments/showcase/geometry_capabilities/03_compiled_world/experiment_blocked.yaml --json
python usage/experiments/showcase/geometry_capabilities/03_compiled_world/replay_valid_route.py
anysearch replay file runtime/geometry_capabilities/03_compiled_world/replay/trajectories/heuristic_astar.json
```

Every command above has been run against the actual code, not sketched --
see each step's README for the verified output.

## Two limitations, stated plainly

- **`anysearch geometry inspect/validate/sample` never reach a native
  provider.** A Python `geometry.py` sibling always takes discovery
  precedence over a compiled native export for the same provider name --
  the same precedence a live training run uses. Step 4's
  `compare_providers.py` calls both providers directly instead; see its
  README for why and what it actually proves.
- **Sparse compiled-world transformations aren't exposed as experiment YAML
  or an `anysearch geometry` command yet** -- there is no config field or CLI
  flag that installs a `SparseBoxTransform` today. Step 3's
  `prepare_world.py` has one clearly labeled section calling that
  programmatic API directly, as a demonstration of what exists, not a claim
  that the CLI performs the operation.
