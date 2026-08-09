# Staged training

Staged training keeps one policy while applying an ordered list of task
configurations. Use it to introduce environment dynamics gradually, for
example by beginning with an adjacent waypoint, increasing the episode budget
and goal distance, and enabling trail placement only in the final stage.

```yaml
staging:
  enabled: true
  resume: true
  replay_transition: clear
  stages:
    - name: select-adjacent-goal
      iterations: 25
      env:
        max_steps: 1
        trail_mode: false
        waypoints_file: path/to/adjacent.json

    - name: full-task-with-trails
      iterations: 200
      env:
        max_steps: 128
        trail_mode: true
        waypoints_file: path/to/full.json
```

Each stage is merged over the shared `env`, `evaluation`, and
`algorithm_config` blocks. `iterations` is the number of iterations spent in
that stage; checkpoint and metric iteration numbers remain cumulative across
the complete run.

## Stage fields

| Field | Default | Purpose |
|---|---:|---|
| `name` | required | Unique stable stage identifier. |
| `iterations` | required | Training iterations allocated to the stage. |
| `env` | `{}` | Environment overrides such as `max_steps`, `trail_mode`, goals, rewards, and task termination. |
| `evaluation` | `{}` | Evaluation overrides for this stage. |
| `algorithm_config` | `{}` | Algorithm hyperparameter overrides for this stage. |
| `replay_transition` | staging default | Selects `clear` or `preserve` when entering this stage. |

`clear` constructs a fresh algorithm and transfers policy weights. This clears
replay and optimizer state, which is appropriate when transition semantics
change, such as enabling trails. `preserve` restores the full preceding
checkpoint, including replay and optimizer state.

## Policy-contract restrictions

Stages cannot change fields that alter policy tensor shapes:

- `agent_count`
- `observation`
- `action`
- `geometry.grid_size`

Create a separate experiment when one of these must change. Runtime task fields
such as episode length, trail behavior, waypoint files, rewards, geometry
contents at a fixed grid size, and termination settings may change between
stages.

The runner writes `staging_state.json` and a checkpoint at each transition.
Resume restores the active stage and continues without repeating completed
stages. MLflow metrics include `training_stage_index`.

See the [progressive waypoint showcase](../../usage/experiments/showcase/staged_waypoint_training/README.md)
for a complete runnable configuration.
