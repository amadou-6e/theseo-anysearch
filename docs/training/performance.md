# Training performance metrics

AnySearch records the timing data already produced by RLlib and its own
post-training phases under the TensorBoard `performance/` namespace. Missing
RLlib fields are omitted, which keeps runs compatible across RLlib versions and
algorithms.

The RLlib metrics are:

- `rllib_iteration_ema_s` and `rllib_training_step_ema_s`
- `sampling_ema_s`, `learner_update_ema_s`, and `sync_weights_ema_s`
- `replay_add_ema_s`, `replay_sample_ema_s`, and
  `replay_update_priorities_ema_s`
- `env_step_ema_s` and `inference_ema_s`
- `env_to_module_connector_ema_s` and `module_to_env_connector_ema_s`
- `training_step_calls`, the number of training-step calls in the iteration
- `rllib_wall_time_s`, the current measured `Algorithm.train()` wall time
- `rllib_training_step_estimated_total_s` and
  `rllib_estimated_residual_s`, estimates formed from the training-step EMA
  and current call count

RLlib's timer values are exponential moving averages per timed call, not
additive phase totals. The `_ema_s` suffix makes that distinction explicit.

AnySearch also measures `anysearch_evaluation_s`,
`anysearch_checkpoint_s`, and `anysearch_reporting_s`. These occur outside
the measured `Algorithm.train()` interval.

Use these measurements together: a high sampling time can be separated from
environment, inference, and connector costs; learner and replay timings expose
training-side pressure; and synchronization timing measures policy-weight
broadcast overhead.
