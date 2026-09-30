# Dijkstra

[`run.yaml`](run.yaml) launches Dijkstra as the main algorithm. It searches the voxel action graph with a zero heuristic, executes the resulting action sequence in the environment, and saves a replayable trajectory.

Use this configuration to verify that a scenario is solvable, establish the shortest-path length, and check that A-star variants return valid paths.

```powershell
anysearch run usage/experiments/heuristics/dijkstra/run.yaml
```
