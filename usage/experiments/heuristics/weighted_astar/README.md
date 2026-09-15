# Weighted A-star

[`run.yaml`](run.yaml) launches weighted A-star with a weight of `1.5`. The planner prioritizes `g + 1.5h`, which generally explores fewer nodes but does not guarantee the shortest possible path.

Use this configuration to evaluate the planning-speed versus path-quality tradeoff. Change `heuristic.weight` in the YAML to control that tradeoff.

```powershell
anysearch run usage/experiments/heuristics/weighted_astar/run.yaml
```
