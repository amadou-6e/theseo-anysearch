# Showcase experiments

These are deliberately small examples for learning the experiment format and checking individual parts of the local workflow.

- [`quick_demo.yaml`](quick_demo.yaml) is the general five-iteration PPO smoke test, including checkpoints, trajectories, and local MLflow tracking.
- [`mlflow_demo.yaml`](mlflow_demo.yaml) focuses on verifying metric and parameter logging to the default SQLite MLflow store.
- [`goal_nav_demo.yaml`](goal_nav_demo.yaml) demonstrates goal-directed PPO navigation around a wall with box observations and distance shaping.
- [`trail_geometry_demo.yaml`](trail_geometry_demo.yaml) demonstrates the dynamic trail task, where agent motion fills voxels until the target fill count is reached.
- [`multi_agent_demo.yaml`](multi_agent_demo.yaml) demonstrates three agents sharing one PPO policy and producing a multi-agent replay.

These configurations prioritize quick feedback over final policy quality.
