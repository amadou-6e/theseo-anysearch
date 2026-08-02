# RLlib algorithm adapters

This package owns algorithm-specific settings and construction. Each adapter
translates validated Anysearch settings into an RLlib AlgorithmConfig and
returns the built RLlib algorithm.

## Modules

- models.py contains validated algorithm settings.
- base.py defines the algorithm configuration contract.
- ppo.py, dqn.py, rainbow.py, sac.py, td3.py, and ddpg.py contain
  single-algorithm adapters.
- multi_voxel_ppo.py contains the multi-agent PPO adapter.

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
