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
  background: trace
output:
  directory: artifacts/explanation
seed: 142
```

Run `anysearch explain dqn-waypoints:4d312abc --request explain.yaml`. Request and scenario files reject unknown fields. CLI attribution and output options override request values; source flags cannot be mixed with `--request`.

The background can be `trace` or `mean` (the mean replay observation), or `zeros`. `trace` is recommended because it represents states the policy encountered.

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

## Interactive UI

Install the optional UI dependencies and launch a checkpoint session:

```powershell
pip install -e ".[explain-ui]"
anysearch explain-ui dqn-waypoints:4d312abc --checkpoint latest
```

The browser interface restores the policy once. It can start from a real initial
observation or load an exact observation JSON or fictional-observation YAML. Use
the X, Y, and Z slice selector to edit individual normalized voxel inputs, and
use the generated sidebar controls to change every scalar or vector field within
its declared bounds. Each valid edit recomputes the action scores, selected
movement vector, safe-action margin, and grouped attribution immediately.

Download the edited scenario YAML to reproduce it later with `anysearch explain`,
or download the current JSON report. UI-created observations remain explicitly
marked `not_environment_validated`; editing a value does not claim that the
environment could naturally produce that combination.
