# Training performance metrics

AnySearch records the timing data already produced by RLlib and its own
post-training phases under the TensorBoard `performance/` namespace. Missing
RLlib fields are omitted, which keeps runs compatible across RLlib versions and
algorithms.

The RLlib metrics are:

- `rllib_iteration_s` and `rllib_training_step_s`
- `sampling_s`, `learner_update_s`, and `sync_weights_s`
- `replay_add_s`, `replay_sample_s`, and `replay_update_priorities_s`
- `env_step_s` and `inference_s`
- `env_to_module_connector_s` and `module_to_env_connector_s`
- `rllib_unaccounted_s`, the measured `Algorithm.train()` wall time not
  represented by RLlib's `training_step` timer

AnySearch also measures `anysearch_evaluation_s`,
`anysearch_checkpoint_s`, and `anysearch_reporting_s`. These occur outside
the measured `Algorithm.train()` interval and therefore are not subtracted
from `rllib_unaccounted_s`.

Use these measurements together: a high sampling time can be separated from
environment, inference, and connector costs; learner and replay timings expose
training-side pressure; and synchronization timing measures policy-weight
broadcast overhead.
