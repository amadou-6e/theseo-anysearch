# Settings package

This package is the authoritative schema for AnySearch YAML settings. Keep validation,
defaults, and user-facing `Field(description=...)` text beside the field they describe.
`loading.py` reads YAML and assembles these domain models; it does not redefine fields.

## Naming convention

Use names for what an object does, not where its implementation originated.

| Suffix or prefix | Meaning | Example |
| --- | --- | --- |
| `Config` | A complete configuration block with behavior and defaults | `RewardConfig` |
| `Selector` | A YAML reference to one named implementation plus parameters | `RewardSelector` |
| `Resolved` | An internal runtime value after lookup and precedence resolution | `ResolvedReward` |
| `BuiltIn` | An internal implementation shipped by AnySearch | `BuiltInReward` |
| `Native` | An internal compiled Rust implementation or ABI adapter | `NativeReward` |

Do not introduce `Custom*Config`. “Custom” describes implementation provenance and becomes
ambiguous when built-in, Python, and Rust implementations share one selectable namespace.

For example:

```yaml
env:
  rewards:
    provider:
      name: segment_countdown_goal
      parameters:
        minimum_reward: 1.0
```

`RewardConfig` owns the complete reward block. `RewardSelector` selects the provider. Runtime
resolution searches the registered implementations and records whether the resolved provider is
built-in, Python, or native Rust. A provider supersedes another implementation only when their
selectable names are identical.

The legacy YAML key `rewards.custom` and Python property `rewards.custom` remain compatibility
aliases during migration. New YAML, documentation, and code must use `provider`.

Apply the same pattern to other extension points:

- `PredicateSelector`
- `OutcomeSelector`
- `TrainingMetricSelector`
- `EvaluationMetricSelector`

## Module boundaries

- `environment/`: environment composition, geometry, observations, actions, and rewards
- `training.py`: training execution and early stopping
- `evaluation.py`: deterministic evaluation and evaluation workers
- `execution.py`: execution-backend settings
- `algorithm.py`: base algorithm/model contracts and compatibility validation
- `root.py`: the composed `Settings` object
- `loading.py`: YAML loading, overrides, and typed resolution
- `compatibility.py`: explicitly temporary migration helpers

`theseo_anysearch.models` is a compatibility shim. New code imports from
`theseo_anysearch.settings` or the narrow domain module.