# Segmented inverse-success curriculum

`experiment.yaml` runs the latest 20-trial empty-grid PPO sweep with inverse-success stage sampling, evaluation every five iterations, and route continuation. Every episode contains an exact 150-action-step route (`0.75 * max_steps`) split into equal-distance segments plus one residual segment.

The native `segment_countdown_goal` reward is sparse. For a segment with exact shortest length `L`, its reward budget starts at `2L` and decreases by one for every action spent on that segment. Reaching it on segment step `s` returns `max(2L - s, 1)`. The segment step counter and budget reset immediately when the next waypoint becomes active; no step, collision, shaping, invalid-action, or construction reward is added.

Compile and launch from the repository root:

```powershell
anysearch compile usage/experiments/tune/waypoint_route_segment_reward
anysearch tune usage/experiments/tune/waypoint_route_segment_reward/experiment.yaml
```