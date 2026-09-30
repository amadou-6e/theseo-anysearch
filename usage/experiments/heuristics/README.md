# Heuristic experiments

These experiments run graph-search planners directly in `VoxelEnv`. They do not build PPO, SAC, or another learned policy. Each successful run writes a replayer-compatible `trajectories/heuristic_<type>.json` file.

## Available planners

- [`dijkstra/run.yaml`](dijkstra/run.yaml) uses zero heuristic cost, guaranteeing a shortest path and providing the correctness baseline for A-star-family planners.
- [`weighted_astar/run.yaml`](weighted_astar/run.yaml) uses `g + 1.5h`, accepting potentially longer paths in exchange for a stronger preference toward the goal and typically faster search.

Both examples use the same cube environment and seed so their path and planning behavior can be compared directly.
