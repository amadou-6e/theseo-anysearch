# RLlib trainer package

The trainer package owns the lifecycle of one configured training run. It does
not define algorithm-specific hyperparameters or construct RLlib algorithms.

## Modules

- base.py defines the small abstract BaseTrainer lifecycle contract.
- trainer.py implements training, evaluation, checkpointing, resume, custom
  metrics, custom rewards, trajectory recording, and early stopping.
- results.py exposes normalized RLlib and project training result models.
- evaluation.py evaluates policies and curriculum stages.
- parallel_evaluation.py configures and executes vectorized evaluation.
- early_stop.py evaluates configured training termination conditions.

The algorithm-named modules in this package are compatibility imports. New code
must import algorithm adapters from theseo_anysearch.rllib.algorithms.

## Dependency direction

    runner/tune -> trainer -> algorithms -> RLlib
                        |
                        +-> evaluation and reporting

A runner selects where a job executes. A trainer controls one job. An algorithm
adapter constructs the RLlib algorithm used by that job. Tune coordinates
multiple jobs.

## Documentation

Public classes, methods, and functions use NumPy-style docstrings. New
parameters belong under Parameters, returned values under Returns, and raised
public exceptions under Raises.
