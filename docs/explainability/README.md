# Explaining policy decisions

`anysearch explain` restores a trained DQN checkpoint and explains why it preferred its recorded action over the best visibly safe alternative. Grouped occlusion replaces one observation group at a time with a background value and measures the change in the Q-value margin.

## Explain a saved trajectory

```powershell
anysearch explain experiments/train/000_example/runtime/run-id --trace best
anysearch explain dqn-waypoints:4d312abc --checkpoint latest --trace iter_000080
```

The run may be a directory or registered `experiment-name:run-id`. `--trace` accepts `best`, `latest`, an iteration filename, or a JSON trajectory path. AnySearch recreates the environment from `experiment.yaml` and replays every action. It stops at the first cursor mismatch and never silently explains a divergent replay.

Select steps using `--focus collisions`, `--focus all --max-steps 20`, or `--focus explicit --steps 0,4,9`.

## Reusable request files

```yaml
checkpoint: latest
source:
  trace: best
explanation:
  method: occlusion
  focus: collisions
  max_steps: 50
  background: auto
output:
  directory: artifacts/explanation
seed: 142
```

Run `anysearch explain dqn-waypoints:4d312abc --request explain.yaml`. Request and scenario files reject unknown fields. CLI attribution and output options override request values; source flags cannot be mixed with `--request`.

The background can be `auto`, `trace`/`mean` (both average every observation in the replayed trace or scenario into a single reference observation), or `zeros`. `auto` uses trace observations for multi-step inputs and zeros for one-step scenarios. `trace`/`mean` is recommended when available because it represents states the policy encountered; `zeros` is a hard occlusion but is *not* semantically neutral for every feature group (0.0 encodes "empty cell" / "no ray hit", not "no information"), so treat zero-background attributions as an occlusion-to-open-space effect rather than an unbiased baseline.

An explicitly requested `trace` or `mean` background requires at least two observations, since a single-observation background collapses to the observation itself and always attributes 0.0 to every group. `auto` prevents this degeneration for single-step and fictional scenarios.

Modern DQN checkpoints restore only their saved RLModule for explanations. This
avoids starting Ray, rebuilding a trainer, or allocating rollout workers merely
to score observations.

## Controlled environment scenarios

```yaml
type: environment
seed: 142
state:
  cursor: [4, 4, 4]
  route:
    - [6, 6, 6]
  geometry_boxes:
    - [8, 8, 0, 8, 8, 31]
  trail: []
execution:
  mode: rollout
  max_steps: 25
```

Run `anysearch explain dqn-waypoints:4d312abc --scenario scenario.yaml --focus all`. Execution modes are `single_step`, `rollout`, and `actions`; the latter takes an explicit `actions` list. These reports are `environment_validated`.

## Fictional observations

Set `type: observation`, `chosen_action: policy` (or an action index), and provide an `observation` mapping containing every field in the restored policy's observation space. Missing or extra fields, wrong shapes, non-finite numbers, and out-of-bounds values are errors. Reports are `not_environment_validated` because the environment did not produce the state.

## Output and interpretation

Artifacts go to `<run>/explanations/<id>/` unless `--output` is set:

- `summary.md`: readable action-margin and strongest-feature summary;
- `report.json`: complete scores, goal features, attributions, and provenance;
- `steps.csv`: compact per-step comparison;
- `observations/step_XXXXXX.json`: exact pre-action network inputs;
- `request.yaml`: fully resolved settings.

Attribution is relative to `chosen Q-value - best-safe Q-value`. A positive group attribution means that group increased preference for the chosen action; a negative value reduced it. Occlusion describes policy sensitivity, not causal correctness.

## Current scope

Checkpoint restoration supports DQN with discrete voxel action spaces. The available method is grouped occlusion. Unsupported algorithms, vector actions, malformed scenarios, and divergent traces raise explicit errors.
