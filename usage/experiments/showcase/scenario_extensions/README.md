# Scenario extensions

This standalone showcase places the agent in the center of an empty grid and
selects one of its 26 adjacent voxels as the goal before each episode reset.
Training samples a seeded direction. Evaluation uses 26 deterministic seeds and
therefore checks every `discrete_26` direction exactly once.

Run the Python provider directly:

```text
anysearch run usage/experiments/showcase/scenario_extensions
```

The sibling `scenarios.py` function name matches
Training selects the Python `adjacent_goal_python` provider from `scenarios.py`.
Evaluation selects the Rust `adjacent_goal_rust` provider from the compiled
extension. This demonstrates that Python and Rust providers can coexist in one
experiment; a Rust provider supersedes Python only when both expose the same
selectable name.

The `extension/` directory implements the same provider in Rust with
`#[anysearch_scenario]`. Compile it and rerun the same experiment to use the
native provider with the identical YAML name:

```text
anysearch compile usage/experiments/showcase/scenario_extensions
anysearch run usage/experiments/showcase/scenario_extensions
```

When both implementations exist, the compiled Rust provider with the selected
name takes precedence. AnySearch never silently falls back after selecting an
invalid or missing implementation.
