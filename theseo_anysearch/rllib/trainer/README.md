# RLlib trainer package

The trainer package owns the lifecycle of one configured training run. It does
not define algorithm-specific hyperparameters or construct RLlib algorithms.

## Structure

    trainer/
    ├── trainer.py
    ├── lifecycle.py
    ├── checkpointing.py
    ├── evaluation/
    │   ├── coordinator.py
    │   ├── evaluator.py
    │   ├── retention.py
    │   ├── generalization.py
    │   └── parallel.py
    ├── curriculum/
    │   ├── controller.py
    │   ├── advancement.py
    │   └── sampling.py
    ├── reporting/
    │   ├── tensorboard.py
    │   ├── metrics.py
    │   └── trajectories.py
    ├── results.py
    └── runtime.py

## Ownership

- trainer.py coordinates collaborators and exposes the public training lifecycle.
- lifecycle.py builds the algorithm and executes timed training iterations.
- checkpointing.py persists and restores RLlib and project state.
- evaluation/coordinator.py runs evaluation and combines its outcomes.
- evaluation/evaluator.py defines normalized success and evaluation metrics.
- evaluation/parallel.py performs vectorized deterministic policy evaluation.
- evaluation/retention.py owns evaluation of previously visited curriculum stages.
- evaluation/generalization.py owns evaluation of unseen curriculum cases.
- curriculum/controller.py owns curriculum state coordination.
- curriculum/advancement.py owns stage-advancement decisions.
- curriculum/sampling.py owns training-stage sampling.
- reporting/tensorboard.py writes run-local TensorBoard events.
- reporting/metrics.py computes Python and native training metrics.
- reporting/trajectories.py records periodic and best trajectory artifacts.
- results.py contains normalized RLlib and project result models.
- runtime.py contains shared Ray and runtime helpers.

The current develop baseline has no curriculum, retention, or generalization
runtime implementation to relocate. Their modules establish ownership without
inventing unused behavior. Curriculum code should enter through these modules
when its branch is merged.

## Dependency direction

    runner/tune -> trainer -> algorithms -> RLlib
                       |
                       +-> evaluation, curriculum, and reporting

A runner selects where a job executes. A trainer coordinates one job. An
algorithm adapter constructs its RLlib algorithm. Tune coordinates multiple
jobs.

## Documentation

Public classes, methods, and functions use NumPy-style docstrings. New
parameters belong under Parameters, returned values under Returns, and raised
public exceptions under Raises.