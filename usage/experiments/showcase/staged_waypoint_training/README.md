# Progressive staged waypoint training

This showcase teaches a single DQN policy in five explicit stages:

1. Select an adjacent goal in a one-step episode with trails disabled.
2. Reach a goal four voxels away with a larger episode budget.
3. Reach a goal twelve voxels away.
4. Train against the full distant goal and full episode budget.
5. Keep the full task and enable trail placement.

The observation and action contracts remain fixed, so weights transfer safely.
Replay is cleared at transitions; this is especially important before the last
stage because enabling trails changes the environment transition function.

Run it from the repository root:

```text
anysearch experiment run usage/experiments/showcase/staged_waypoint_training/experiment.yaml
```

The run directory contains `staging_state.json`, cumulative checkpoints, and
stage-indexed MLflow metrics.
