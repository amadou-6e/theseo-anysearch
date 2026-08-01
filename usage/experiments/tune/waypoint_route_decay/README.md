# Time-decayed waypoint-route reward

This 20-trial Tune experiment uses an obstacle-free `discrete_18` waypoint route curriculum and a compiled Rust reward named `time_decayed_goal`.

The YAML sets `goal_reward: 150.0`, matching the 150-step route length. The extension replaces every built-in reward:

```text
non-goal step: 0
waypoint reached at episode step t: max(1, 150 - t)
```

The countdown is episode-global and does not reset at intermediate waypoints. This rewards completing every waypoint while preferring faster progress through the entire route. Step, collision, invalid-action, distance-shaping, and construction rewards are all zero.

Compile before launching the sweep:

```powershell
anysearch compile usage/experiments/tune/waypoint_route_decay
anysearch tune --config usage/experiments/tune/waypoint_route_decay/experiment.yaml
```