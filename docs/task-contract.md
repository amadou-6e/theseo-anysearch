# Task, goal, reward, and termination contract

Voxel tasks use a versioned `env.task` block. Existing experiment YAML files
that omit it retain exact point-goal navigation.

```yaml
env:
  max_steps: 200
  waypoints_file: usage/experiments/train/tiny_overfit_waypoints.json
  rewards:
    step_cost: -0.01
    distance_shaping: 0.2
    goal_reward: 1.0
    collision_cost: -0.1
    invalid_action_cost: -0.1
    construction_residual_weight: 0.0
    construction_overshoot_weight: 0.0
  task:
    version: 1
    goal:
      type: point
      position: [6, 7, 1]
      tolerance: 0.0
    termination:
      terminate_on_success: true
```

For a region-like goal, use `type: target_voxel_set` and provide `voxels`.
The first voxel supplies goal-directed observations, while entering any member
satisfies the task predicate. An explicit task goal requires a waypoint file so
the start position is unambiguous.

Every step returns these `info` fields:

- `goal_reached` and `termination_reason` (`in_progress`, `success`, or
  `step_limit`);
- `reward_breakdown`, with step, distance, success, invalid-action, collision,
  construction-residual, and construction-overshoot components;
- `unshaped_reward`, which excludes distance shaping;
- initial, final, and minimum goal distance.

Episode trajectories retain the same breakdown and distances. Evaluation
metrics expose component means as `evaluation_reward_<component>_mean`, so Tune,
TensorBoard, MLflow, and CLI reporters receive the same attribution.

Distance-progress shaping is potential based:

```text
distance_shaping * (previous Euclidean distance - current Euclidean distance)
```

Set `distance_shaping: 0.0` to disable it. Construction weights are non-negative;
their reward terms are the negative weighted residual and overshoot voxel counts.
