# Canonical training and evaluation metrics

AnySearch separates task outcomes, waypoint progress, curriculum state,
optimization health, and runtime performance. TensorBoard and MLflow use the
same slash-delimited names for new runs.

General task evaluation uses `eval/task/episodes`, `success_rate`,
`return_mean`, `episode_len_mean`, `steps_to_success_mean`, `collision_rate`,
`termination_rate`, and `truncation_rate`. Distance metrics remain available
for point-goal tasks. Unshaped return is emitted only when it differs from the
actual return; reward components are emitted only when multiple components
exist. This avoids aliases such as finish rate versus success rate.

Waypoint metric providers are grouped under `eval/task/waypoint/`. Training
environment-runner aggregates use `train/task/waypoint/`. A metric is omitted
when RLlib does not provide it; missing data is never reported as zero.

Curriculum retention evaluations expose:

```text
curriculum/stage
curriculum/retention_success_rate
curriculum/stage_<n>/success_rate
curriculum/stage_<n>/completion_fraction
curriculum/stage_<n>/sampling_probability
```

The exact episode results remain in `evaluation/curriculum_iter_*.json`.
Transition and pass pulses are not emitted because immediate-success curricula
encode the same event in changes to `curriculum/stage`.

Available RLlib learner fields are mapped to `train/optimization/`, including
PPO policy/value loss, entropy, KL, clipping, explained variance, learning rate,
and gradient norm, or DQN TD loss, Q values, exploration epsilon, and replay
buffer size. Algorithm-inapplicable fields are omitted rather than creating
empty charts.
