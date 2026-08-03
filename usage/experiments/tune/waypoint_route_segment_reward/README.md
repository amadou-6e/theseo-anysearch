# Segmented inverse-success curriculum

`experiment.yaml` runs the latest 20-trial empty-grid PPO sweep with inverse-success stage sampling, evaluation every five iterations, and route continuation. Every episode contains an exact 72-action-step route (`0.75 * 96`) split into equal-distance segments plus one residual segment. The standalone PPO and DQN configurations use fixed 96-step routes.

The native `segment_countdown_goal` reward is sparse. For a segment with exact shortest length `L`, its reward budget starts at `budget_multiplier * L` and decreases by one for every action spent on that segment. Reaching it on segment step `s` returns `max(budget_multiplier * L - s, minimum_reward)`. The segment step counter and budget reset immediately when the next waypoint becomes active; no step, collision, shaping, invalid-action, or construction reward is added.

The native extension also reports two evaluation metrics:

- `evaluation_waypoints_reached`: total intermediate and final waypoints reached across the evaluation episodes.
- `evaluation_waypoint_completion_fraction`: reached waypoints divided by configured route waypoints across the evaluation episodes.

Compile and launch from the repository root:

```powershell
anysearch compile usage/experiments/tune/waypoint_route_segment_reward
anysearch tune usage/experiments/tune/waypoint_route_segment_reward/experiment.yaml
```
