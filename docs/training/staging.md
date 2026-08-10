# Staged training

Staging trains one policy through an ordered list of tasks. Each stage may
override environment, evaluation, and algorithm settings without changing the
policy's observation or action contract.

```yaml
staging:
  enabled: true
  resume: true
  replay_transition: clear
  stages:
    - name: select-adjacent-goal
      completion:
        type: iterations
        iterations: 25
      env:
        max_steps: 1
        trail_mode: false

    - name: navigate-with-trails
      completion:
        type: any
        max_iterations: 200
        conditions:
          - type: performance
            metric: evaluation_success_rate
            threshold: 0.95
            comparison: gte
            consecutive_iterations: 5
      env:
        max_steps: 128
        trail_mode: true
```

## Completion conditions

`completion` is a condition tree. The built-in condition types are:

| Type | Configuration | Meaning |
| --- | --- | --- |
| `iterations` | `iterations` | Complete after this many stage-local iterations. |
| `performance` | `metric`, `threshold`, `comparison`, `consecutive_iterations` | Complete after a metric satisfies a threshold for consecutive iterations. |
| `all` | `conditions` | Complete when every child condition is true. |
| `any` | `conditions` | Complete when at least one child condition is true. |
| `not` | `condition` | Negate one child condition. |
| `python` | `callable`, optional `parameters` | Delegate a leaf condition to Python. |

Comparisons are `gte`, `gt`, `lte`, `lt`, and `eq`. Metrics come from the
normalized `TrainResult.standard_metrics()` contract, including
`episode_reward_mean`, `training_success_rate`, and
`evaluation_success_rate`.

Conditions can be nested:

```yaml
completion:
  type: any
  conditions:
    - type: all
      conditions:
        - type: performance
          metric: evaluation_success_rate
          threshold: 0.95
          consecutive_iterations: 5
        - type: performance
          metric: episode_len_mean
          comparison: lte
          threshold: 32
    - type: iterations
      iterations: 200
```

Every stage must have a finite upper bound. An `iterations` condition can
provide that bound when its position in the tree guarantees termination.
Alternatively, set `max_iterations` on the root condition. The cap completes
the stage even when the configured condition never becomes true.

## Python extension conditions

A Python leaf references an importable callable with
`module.path:function_name`:

```yaml
completion:
  type: python
  callable: my_training.completion:ready_for_trails
  max_iterations: 200
  parameters:
    minimum_success: 0.9
```

The callable receives one context dictionary and returns a truthy value when
the leaf is complete:

```python
def ready_for_trails(context):
    success = context["metrics"]["evaluation_success_rate"]
    state = context["state"]
    required = context["parameters"]["minimum_success"]
    state["best_success"] = max(state.get("best_success", 0.0), success)
    return state["best_success"] >= required
```

The context contains `result` (the normalized `TrainResult`), `metrics`,
`stage_iteration`, `parameters`, and a mutable `state` dictionary. State must
remain JSON-serializable; it is persisted in `staging_state.json` after every
iteration and restored on resume. Python leaves compose with `all`, `any`, and
`not` exactly like built-in leaves. The referenced module must be importable in
the training process.

## Stage transitions and resume

`replay_transition: clear` creates a fresh algorithm and transfers policy
weights, resetting replay and optimizer state. `preserve` restores the full
checkpoint. A stage can override the staging default.

The runner checkpoints at transitions and writes `staging_state.json`, which
contains the active stage, its original start iteration, and condition state.
`resume: true` allows a run to continue inside the active stage without
resetting consecutive-performance or Python-extension state.

Stage overrides may not change `agent_count`, `observation`, `action`, or
`geometry.grid_size`, because those fields define the policy contract.
