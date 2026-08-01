# Custom training and evaluation metrics

Experiments may define arbitrary numeric metrics in Python without adding a YAML import path or changing the framework. The runner discovers modules beside the experiment YAML by convention.

For `example.yaml`, the lookup order is:

1. `evaluation_metrics.example.py`, then `evaluation_metrics.py`
2. `training_metrics.example.py`, then `training_metrics.py`

An experiment-specific module replaces the shared fallback for that scope. Each module defines one function:

```python
def compute_metrics(context):
    return {"navigation_score": 10.0 * context.standard_metrics["evaluation_success_rate"]}
```

Returned names are automatically prefixed. The example above becomes `evaluation_navigation_score`; a training metric named `sample_efficiency` becomes `training_sample_efficiency`. Values must be finite numbers, names must be Python identifiers, and names may not replace built-in metrics.

## Contexts

`EvaluationContext` provides:

- `iteration`
- `episodes`, including trajectories and preserved `final_info`
- `standard_metrics`
- `env_config`
- `final_infos`

`TrainingContext` provides:

- `iteration`
- `standard_metrics`
- the raw `rllib_result`
- `environment_steps_total`
- `duration_s`
- `env_config`

Custom values use the existing scalar reporting contract, so they appear in Tune and ASHA payloads, TensorBoard, MLflow, Tune history, and run artifacts. The discovered source files are copied into each run or Tune trial, and `custom_metrics.json` records their SHA-256 hashes.

## ASHA navigation-score example

[`evaluation_metrics.ppo_asha.py`](../../usage/experiments/tune/evaluation_metrics.ppo_asha.py) ranks policies using successful finishes first and normalized goal progress second:

```text
evaluation_navigation_score = 10 * success_rate + clipped_progress_fraction_mean
```

The adjacent `ppo_asha.yaml` selects this metric directly:

```yaml
tune_config:
  metric: evaluation_navigation_score
  mode: max
```

No metric module path is configured in YAML; adjacency and the YAML stem determine discovery.
