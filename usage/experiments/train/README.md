# Training experiments

These configurations are longer-running policy-training baselines and task-specific benchmarks.

## General and geometry-specific baselines

- [`ppo_baseline.yaml`](ppo_baseline.yaml) is the standard four-agent PPO reference on stepped terrain.
- [`ppo_corridor.yaml`](ppo_corridor.yaml) targets two-agent navigation through a constrained L-shaped corridor with a smaller encoder.
- [`ppo_spiral.yaml`](ppo_spiral.yaml) targets a harder vertical spiral-ramp task with longer episodes and a deeper encoder.
- [`sac_baseline.yaml`](sac_baseline.yaml) is the off-policy SAC reference on the pipe-junction geometry.
- [`ppo_diverse.yaml`](ppo_diverse.yaml) trains PPO on high-resolution geometry-pool samples with aggressive obstacle augmentation.
- [`r3_highres_finetune.yaml`](r3_highres_finetune.yaml) fine-tunes a pretrained radius-3 encoder on high-resolution stepped terrain.

## Map and action-space comparisons

- [`ppo_maps.yaml`](ppo_maps.yaml) is the discrete 26-action radial-observation baseline on augmented industrial map crops.
- [`ppo_maps_zones.yaml`](ppo_maps_zones.yaml) replaces progress shaping with always-negative zone-based distance rewards.
- [`dqn_maps_zones.yaml`](dqn_maps_zones.yaml) applies the same map and zone-reward task to off-policy DQN.
- APPO curriculum examples live under `../tune/waypoint_route_segment_reward/` because they exercise asynchronous sampling together with curriculum evaluation.
- [`ppo_maps_vector_zones.yaml`](ppo_maps_vector_zones.yaml) replaces discrete actions with a compact three-component action vector.
- [`ppo_maps_vector_zones_long.yaml`](ppo_maps_vector_zones_long.yaml) extends that vector-action experiment to 1,000 iterations.

## Reproduction and diagnostics

- [`ppo_tiny_overfit.yaml`](ppo_tiny_overfit.yaml) is an intentionally tiny fixed task used to diagnose whether PPO, evaluation, and replay can learn and report a simple solution.

Use the showcase configurations for fast installation checks; use these files when comparing learning behavior or producing durable checkpoints.
