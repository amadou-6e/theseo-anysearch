# Heuristic imitation pretraining

The imitation stage teaches the PPO policy to copy successful heuristic actions
before normal reinforcement learning begins:

```text
heuristic demonstrations -> behavior cloning -> PPO policy -> normal PPO
```

Enable it with a top-level `imitation` block. `training.algorithm` remains
`ppo`; the heuristic runs only as the episode-generation provider.

## Configuration

- `enabled` runs imitation before PPO's first iteration.
- `strategy` is currently `pretrain_then_rl`.
- `generation.provider` selects the episode-generation provider, either as
  shorthand (`provider: astar`) or the full selector block
  (`provider: {name: astar, parameters: {...}}`). Built-in names are `astar`,
  `dijkstra`, `weighted_astar`, and `replanning_astar`.
- `generation.provider.parameters.weight` configures `weighted_astar` and is
  not accepted by the other built-in providers.
- `generation.episodes` is the required number of accepted demonstrations.
- `generation.max_attempts` bounds failed or unsolved collection attempts.
- `generation.require_success` discards episodes that do not reach the goal.
- `collection.seed_start` is the first deterministic environment reset seed.
- `collection.validation_fraction` reserves complete episodes for validation.
- `collection.reuse_dataset` reuses existing data only when its fingerprint
  matches every environment, task, observation, action, and generation
  provider setting.
- `collection.dataset_dir` optionally points multiple runs or Tune trials to
  one shared compatible dataset. Concurrent collectors coordinate by dataset
  fingerprint; incompatible environment or generation provider settings are
  regenerated.
- `collection.curriculum_stages` selects `initial` (the default) or `all`.
  `all` collects demonstrations round-robin across every configured
  `continue_route` segment distance, including a capped final stage when the
  maximum is not an exact increment. Every demonstration receives a new,
  deterministically seeded route; duplicate routes are rejected. This keeps
  the dataset balanced across curriculum difficulty without repeatedly
  training on one fixed route per stage.
  Successful episodes are collected against an exact per-stage quota, so
  rejected generation-provider rollouts cannot silently underrepresent a
  difficult stage.
- `sampling.provider` selects the batch-sampling provider used during
  pretraining. It defaults to `uniform_transition`, which samples individual
  demonstration transitions uniformly at random. `uniform_episode` instead
  samples whole episodes uniformly, then draws transitions from within each
  sampled episode, which reduces the influence of long episodes relative to
  short ones. Configure it the same way as `generation.provider`, either as
  shorthand (`provider: uniform_episode`) or the full selector block.
- `pretraining.epochs` is the maximum number of behavior-cloning passes.
- `pretraining.batch_size` is the supervised optimizer batch size.
- `pretraining.learning_rate` applies only to behavior cloning.
- `pretraining.label_smoothing` configures categorical cross-entropy smoothing.
- `pretraining.early_stopping_patience` stops on stagnant validation loss.
- `handoff.initialize_encoder` retains learned observation features.
- `handoff.initialize_policy` retains learned policy/action-head parameters.
- `handoff.initialize_value_head` retains value parameters and defaults to
  false because action labels do not supervise values.
- `cache.enabled` reuses a content-addressed behavior-cloned checkpoint.
- `cache.directory` overrides the cache root. Tune automatically enables an
  experiment-level cache when imitation is enabled.
- `cache.refresh` retrains and replaces the matching entry.
- `cache.lock_timeout_seconds` bounds how long a parallel trial waits for the
  trial currently publishing the same network.

## Tuning and cache identity

Tune trials with the same demonstration, network, optimizer, and handoff
contract pretrain once. Trials waiting for that contract load the atomically
published checkpoint. PPO-only settings such as learning rate, gamma, lambda,
clipping, KL coefficient, or RL batch sizes do not affect the key.

The key changes with model structure and tensor shapes, observation and action
contracts, geometry and curriculum inputs, generation and collection settings,
behavior-cloning optimizer settings, handoff settings, or policy ID. Trial IDs,
output directories, timestamps, rollout seed offsets, and copied native
extension paths are excluded. Cache hit/miss, validation metrics, and the cache
key are written to trial artifacts and tracking metrics.

Heterogeneous policy IDs receive independent keys, so differently configured
models cannot consume each other's checkpoints. Agents mapped to the same
shared policy ID reuse that policy's artifact.

Demonstrations record the policy observation before each generation-provider
action. The dataset is split by episode, not by step. Artifacts are written
under the run's `imitation` directory: compressed data, manifest, epoch
metrics, result, and policy checkpoint.

## Python generation providers

Place Python generation providers in `imitation.py` beside the experiment
YAML, following the same convention as scenario providers in
`docs/extensions/scenarios.md`. Each provider is a function named after its
`generation.provider.name` that accepts one `EpisodeGenerationContext` and
returns a demonstration episode; AnySearch discovers it by name from the
sibling file and archives the exact source used with the run. Built-in
provider names (`astar`, `dijkstra`, `weighted_astar`, `replanning_astar`) are
reserved and cannot be shadowed by a Python provider of the same name. If a
future Rust generation provider exposes the identical selected name, Rust
would supersede Python, mirroring scenario provider resolution — but Rust
generation providers are not yet supported, so today every generation
provider outside the built-ins is Python.

Use `usage/experiments/train/ppo_tiny_overfit_imitation.yaml` for the first
validation and compare it with `ppo_tiny_overfit.yaml`.
