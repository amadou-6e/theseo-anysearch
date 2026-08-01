# Tuning experiments

These files launch multi-trial searches or repeat one configuration across multiple geometries. They require substantially more compute than the showcase files.

## PPO searches

- [`ppo_asha.yaml`](ppo_asha.yaml) tunes core PPO optimization parameters and prunes weak trials with ASHA.
- [`ppo_arch_search.yaml`](ppo_arch_search.yaml) jointly searches PPO settings and fully connected network depth and width.
- [`ppo_maps_asha.yaml`](ppo_maps_asha.yaml) tunes radial-observation PPO on augmented industrial map crops.
- [`ppo_diverse_asha.yaml`](ppo_diverse_asha.yaml) tunes PPO across augmented high-resolution geometry-pool samples.
- [`multi_agent_ppo_asha.yaml`](multi_agent_ppo_asha.yaml) applies ASHA to shared-policy multi-agent PPO.
- [`multi_agent_ppo_asha_pretrained.yaml`](multi_agent_ppo_asha_pretrained.yaml) repeats the multi-agent search with a pretrained box encoder.
- [`ppo_pbt.yaml`](ppo_pbt.yaml) evolves PPO configurations with population-based training and checkpoint transfer.

## Other comparisons

- [`sac_asha.yaml`](sac_asha.yaml) tunes the off-policy SAC baseline, including target updates and n-step returns.
- [`ppo_sweep_geometries.yaml`](ppo_sweep_geometries.yaml) keeps PPO settings fixed while comparing performance across geometries.
- [`ppo_empty_grid_waypoint_curriculum.yaml`](ppo_empty_grid_waypoint_curriculum.yaml) runs the expanded 20-trial ASHA curriculum sweep on an obstacle-free grid with vectorized training and evaluation.

ASHA is appropriate for independent trials with early stopping. PBT is appropriate when strong trials should transfer weights and mutate their settings during training. The geometry sweep isolates environment difficulty rather than hyperparameter quality.

Tune reports, checkpoint/resume behavior, pruning artifacts, and explicit stop budgets are documented in [Tune lifecycle and trial budgets](../../../docs/tuning/lifecycle.md).

## Custom ranking metrics

`ppo_asha.yaml` automatically discovers both experiment-specific examples:

- `evaluation_metrics.ppo_asha.py` emits `evaluation_navigation_score`, combining success rate with normalized goal progress so ASHA can distinguish policies before the first finish.
- `training_metrics.ppo_asha.py` emits `training_reward_per_episode_step` and `training_episodes_per_1000_env_steps`, demonstrating metrics derived from standard training results and sampled environment steps.
- [`waypoint_route_decay/experiment.yaml`](waypoint_route_decay/experiment.yaml) tests a multi-waypoint empty-grid curriculum with an episode-global sparse countdown reward implemented as a native Rust extension.
