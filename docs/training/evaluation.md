# Goal-finding evaluation

Every training iteration evaluates the current policy with exploration disabled
in fresh environments. The batch always starts at
training.evaluation_seed (default 42) and contains
training.evaluation_episodes consecutive seeds, so iterations are directly
comparable.

training.evaluation_min_success_rate (default 0.5) is the release threshold. A
policy is classified as solved only when the full evaluation batch meets that
threshold. A policy that improves goal distance but records zero successes is
classified as approaching_not_solved, never as solved.

CLI/run results, Tune, MLflow, TensorBoard, and per-iteration JSON summaries use
the same scalar metric map. It includes success count and rate, steps-to-goal
minimum/mean/maximum, final and minimum Euclidean goal distance,
terminated/truncated counts, shaped return, and reconstructed unshaped task
return.

## Evaluation-driven early stopping

`training.early_stop` consumes these deterministic evaluation batches. Reward
and goal-finish conditions use the batch mean reward and completed-goal count.
Heuristic accuracy compares the policy and configured heuristic on the same
seed sequence. Training data and exploratory rollouts never trigger a stop.