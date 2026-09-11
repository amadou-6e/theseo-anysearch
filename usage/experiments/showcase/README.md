# Showcase experiments

These are deliberately small examples for learning the experiment format and checking individual parts of the local workflow.

- [`quick_demo.yaml`](quick_demo.yaml) is the general five-iteration PPO smoke test, including checkpoints, trajectories, and local MLflow tracking.
- [`mlflow_demo.yaml`](mlflow_demo.yaml) focuses on verifying metric and parameter logging to the default SQLite MLflow store.
- [`goal_nav_demo.yaml`](goal_nav_demo.yaml) demonstrates goal-directed PPO navigation around a wall with box observations and distance shaping.
- [`trail_geometry_demo.yaml`](trail_geometry_demo.yaml) demonstrates the dynamic trail task, where agent motion fills voxels until the target fill count is reached.
- [`multi_agent_demo.yaml`](multi_agent_demo.yaml) demonstrates three agents sharing one PPO policy and producing a multi-agent replay.
- [staged_waypoint_training/](staged_waypoint_training/README.md) progressively increases waypoint distance and episode length before enabling trail placement.
- [hunter/](hunter/README.md) demonstrates a compiled Rust predicate and outcome that make one hunter cursor move exactly two voxels per action.
- [hunter_and_hunted/](hunter_and_hunted/README.md) demonstrates ordered heterogeneous agents, independent policies, capture termination, and asymmetric rewards.

These configurations prioritize quick feedback over final policy quality.

## Custom reward example

`quick_demo.yaml` selects `provider.name: quick_demo` and configures `diagonal_penalty_per_extra_axis` under `provider.parameters`. The name resolves the identically named function in `rewards.py`, where the value is read through `context.parameters`. The module adds a small non-positive diagonal-movement penalty to the standard YAML reward and records it as `diagonal_move_penalty` in the reward breakdown.
