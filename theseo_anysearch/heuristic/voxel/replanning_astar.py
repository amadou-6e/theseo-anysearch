"""A* replanning for environments whose traversability changes each step."""

from theseo_anysearch.heuristic.models import VoxelOraclePlan, VoxelOracleReplay
from theseo_anysearch.heuristic.voxel.astar import VoxelAStarOracle


class VoxelReplanningAStarHeuristic(VoxelAStarOracle):
    """Recompute A* from the current environment state before every action."""

    def replay(self, plan: VoxelOraclePlan | None = None) -> VoxelOracleReplay:
        """Replan and execute one action at a time until the episode ends."""
        del plan
        rust_env = self.env._rust_env
        raw_goal = rust_env.goal_pos()
        if raw_goal is None:
            raise ValueError("Voxel environment has no goal after reset")
        goal = self._position(raw_goal)
        actual_positions = [self._position(rust_env.cursor_pos())]
        rewards: list[float] = []
        action_indices: list[int] = []
        terminated = False
        truncated = False
        goal_reached = actual_positions[0] == goal

        while not goal_reached and not terminated and not truncated:
            current_plan = self.plan()
            if not current_plan.action_indices:
                break
            action_index = current_plan.action_indices[0]
            action_indices.append(action_index)
            _, reward, terminated, truncated, info = self.env.step(action_index)
            rewards.append(float(reward))
            actual = self._position(rust_env.cursor_pos())
            actual_positions.append(actual)
            expected = current_plan.positions[1]
            if actual != expected:
                return VoxelOracleReplay(
                    goal_reached=False,
                    terminated=terminated,
                    truncated=truncated,
                    steps_executed=len(rewards),
                    positions=tuple(actual_positions),
                    rewards=tuple(rewards),
                    action_indices=tuple(action_indices),
                    mismatch=(
                        f"Step {len(rewards)}: expected {expected}, got {actual}"
                    ),
                )
            goal_reached = bool(info.get("goal_reached", actual == goal))

        mismatch = None
        if not goal_reached:
            mismatch = "Replanning ended without reaching the goal"
        return VoxelOracleReplay(
            goal_reached=goal_reached,
            terminated=terminated,
            truncated=truncated,
            steps_executed=len(rewards),
            positions=tuple(actual_positions),
            rewards=tuple(rewards),
            action_indices=tuple(action_indices),
            mismatch=mismatch,
        )


__all__ = ["VoxelReplanningAStarHeuristic"]
