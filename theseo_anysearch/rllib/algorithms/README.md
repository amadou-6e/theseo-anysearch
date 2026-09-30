# RLlib algorithm adapters

This package owns algorithm-specific settings and construction. Each adapter
translates validated Anysearch settings into an RLlib AlgorithmConfig and
returns the built RLlib algorithm.

## Modules

- models.py contains validated algorithm settings.
- base.py defines the algorithm configuration contract.
- ppo.py, dqn.py, rainbow.py, sac.py, td3.py, and ddpg.py contain
  single-algorithm adapters.
- ppo.py contains both the single-agent and shared-policy multi-agent PPO adapters.

Algorithm adapters may configure models, rollout workers, evaluation workers,
resources, and the registered environment. They must not own the training
iteration loop, checkpoint policy, early stopping, or reporting; those belong
to the trainer package.

## Adding an algorithm

1. Add its validated settings model to models.py.
2. Implement its adapter in a module named after the YAML algorithm value.
3. Derive the adapter from Trainer and implement _build_algorithm.
4. Export it from this package.
5. Add construction and lifecycle tests.

Public APIs use NumPy-style docstrings.
## RLlib execution stacks

The supported single-agent PPO, DQN, and Rainbow adapters use RLlib's current
RLModule/Learner and EnvRunner/Connector V2 stacks. Structured AnySearch
observations are flattened by an env-to-module connector, so the public YAML
observation format remains unchanged. Dedicated evaluation EnvRunners receive
weights from the LearnerGroup.

Temporary exceptions are explicit:

- SAC remains on RLlib's legacy stack until its discrete-action model and
  replay configuration are migrated and covered by a real training test.
- Multi-agent voxel PPO remains on the legacy shared-Policy path until it is
  represented as a MultiRLModule.
- DDPG and TD3 are unavailable because current AnySearch environments do not
  expose their required continuous action space.
- TorchModelV2 custom CNNs are rejected by modern-stack adapters until they are
  implemented as RLModules. Standard fully connected models are supported.
- APPO is not currently a registered AnySearch algorithm adapter.
## DQN replay and Learner resources

DQN replay behavior and Learner placement are YAML-selectable:

```yaml
training:
  num_learners: 1
  num_cpus_per_learner: 1
  num_gpus_per_learner: 0.3333333333333333
  weight_sync_interval: 4

algorithm_config:
  replay_buffer_type: uniform  # uniform or prioritized
  replay_buffer_capacity: 200000
```

`uniform` uses RLlib's `EpisodeReplayBuffer`; `prioritized` uses
`PrioritizedEpisodeReplayBuffer`. DQN defaults to uniform replay to preserve
its historical behavior. Rainbow defaults to prioritized replay. When
`num_gpus_per_learner` is omitted, `training.num_gpus` remains the backward-
compatible GPU allocation.

Uniform replay omits per-sample TD-error transfer to the driver because no
priority update consumes it. `training.weight_sync_interval` controls how
often DQN broadcasts learner weights to EnvRunners; its default of `1`
preserves synchronization after every DQN training step.
