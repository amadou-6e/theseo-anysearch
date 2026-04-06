from __future__ import annotations

import io
import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from theseo_anysearch.experiments.output import OutputStore


# ---------------------------------------------------------------------------
# Data classes — filled in by collect_eval_episode(); also usable directly
# ---------------------------------------------------------------------------

@dataclass
class VoxelStepData:
    step: int
    action: int        # 0..25 = index into 26-neighbor cube {-1,0,1}³ \ origin
    reward: float
    done: bool
    cursor_x: int      # cursor position AFTER the step (destination for movement)
    cursor_y: int
    cursor_z: int
    voxel_count: int   # total filled voxels AFTER this step
    placed: bool       # True if a new voxel was filled this step (Place or trail)


@dataclass
class VoxelEpisodeData:
    agent_count: int
    max_steps: int
    obs_mode: str
    init_filled: list[tuple[int, int, int]]  # filled voxels at episode start
    steps: list[VoxelStepData]
    total_reward: float
    success: bool      # True if episode ended before max_steps (reached target)
    grid_size: int = 32
    start_pos: tuple[int, int, int] | None = None
    goal_pos: tuple[int, int, int] | None = None


# ---------------------------------------------------------------------------
# Eval episode collector
# ---------------------------------------------------------------------------

def collect_eval_episode(algo: Any, env_config: dict, *, env: Any = None, seed: int | None = None) -> VoxelEpisodeData:
    """
    Run one evaluation episode using the given RLlib algorithm's policy and
    return the trajectory as a VoxelEpisodeData.

    Creates a fresh VoxelEnv directly (not through Ray workers) so it runs
    in the main process.  Falls back to random actions if compute_single_action
    is unavailable.

    Raises ImportError if theseo_core is not installed.

    ``env`` may be supplied explicitly (e.g. in tests) to bypass VoxelEnv construction.
    """
    if env is None:
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
        env = VoxelEnv(env_config)
    obs, _ = env.reset(seed=seed)

    init_filled: list[tuple[int, int, int]] = []
    start_pos: tuple[int, int, int] | None = None
    goal_pos: tuple[int, int, int] | None = None
    if hasattr(env, "_rust_env") and env._rust_env is not None:
        init_filled = [
            (int(x), int(y), int(z))
            for x, y, z in env._rust_env.filled_voxels()
        ]
        raw_goal = env._rust_env.goal_pos()
        if raw_goal is not None:
            goal_pos = (int(raw_goal[0]), int(raw_goal[1]), int(raw_goal[2]))
        raw_cursor = env._rust_env.cursor_pos()
        start_pos = (int(raw_cursor[0]), int(raw_cursor[1]), int(raw_cursor[2]))

    steps: list[VoxelStepData] = []
    total_reward = 0.0
    step_count = 0
    prev_voxel_count = _extract_voxel_count(obs)

    while True:
        try:
            action = int(algo.compute_single_action(obs, policy_id="default_policy"))
        except Exception:
            action = 0  # fallback: move +X

        obs_next, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        voxel_count = _extract_voxel_count(obs_next)
        placed = voxel_count > prev_voxel_count

        # Record cursor position AFTER the step: for movement actions this is
        # the destination cell where any trail voxel was placed.
        cursor = (1, 1, 1)
        if hasattr(env, "_rust_env") and env._rust_env is not None:
            cursor = env._rust_env.cursor_pos()

        steps.append(VoxelStepData(
            step=step_count,
            action=action,
            reward=float(reward),
            done=done,
            cursor_x=int(cursor[0]),
            cursor_y=int(cursor[1]),
            cursor_z=int(cursor[2]),
            voxel_count=voxel_count,
            placed=placed,
        ))

        total_reward += float(reward)
        prev_voxel_count = voxel_count
        step_count += 1

        if done:
            break
        obs = obs_next

    try:
        env.close()
    except Exception:
        pass

    max_steps = env_config.get("max_steps", 200)
    success = step_count < max_steps  # ended before the time limit

    return VoxelEpisodeData(
        agent_count=1,
        max_steps=max_steps,
        obs_mode=env_config.get("obs_mode", "scalar"),
        grid_size=env_config.get("grid_size", 32),
        init_filled=init_filled,
        steps=steps,
        total_reward=total_reward,
        success=success,
        start_pos=start_pos,
        goal_pos=goal_pos,
    )


# ---------------------------------------------------------------------------
# Multi-agent data classes
# ---------------------------------------------------------------------------

@dataclass
class MultiVoxelStepData:
    step: int
    actions: list[int]           # per-agent, len == agent_count
    rewards: list[float]         # per-agent
    done: bool
    cursors: list[tuple[int, int, int]]   # per-agent cursor AFTER step
    placed: list[bool]           # per-agent: True if a new voxel was filled


@dataclass
class MultiVoxelEpisodeData:
    agent_count: int
    max_steps: int
    steps: list[MultiVoxelStepData]
    total_rewards: list[float]   # per-agent cumulative
    start_positions: list[tuple[int, int, int] | None]
    goal_positions: list[tuple[int, int, int] | None]
    init_filled: list[tuple[int, int, int]]  # geometry voxels at episode start


def collect_multi_eval_episode(
    algo: Any,
    env_config: dict,
    *,
    env: Any = None,
    seed: int | None = None,
) -> MultiVoxelEpisodeData:
    """
    Run one evaluation episode using the multi-agent env and trained shared policy.
    Returns a MultiVoxelEpisodeData with per-agent per-step data.
    """
    if env is None:
        from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv
        env = MultiVoxelEnv(env_config)

    obs, _ = env.reset(seed=seed)
    agent_count = len(env.possible_agents)

    init_filled: list[tuple[int, int, int]] = []
    start_positions: list[tuple[int, int, int] | None] = [None] * agent_count
    goal_positions:  list[tuple[int, int, int] | None] = [None] * agent_count
    if hasattr(env, "_rust_env") and env._rust_env is not None:
        init_filled = [(int(x), int(y), int(z)) for x, y, z in env._rust_env.filled_voxels()]
        for i, pos in enumerate(env._rust_env.cursor_positions()):
            start_positions[i] = (int(pos[0]), int(pos[1]), int(pos[2]))
        for i, pos in enumerate(env._rust_env.goal_positions()):
            if pos is not None:
                goal_positions[i] = (int(pos[0]), int(pos[1]), int(pos[2]))

    max_steps = env_config.get("max_steps", 200)

    steps: list[MultiVoxelStepData] = []
    total_rewards = [0.0] * agent_count
    prev_cursor = {a: start_positions[i] for i, a in enumerate(env.possible_agents)}

    step_count = 0
    while env.agents:
        actions: dict[str, int] = {}
        for agent_id in env.agents:
            agent_obs = obs[agent_id]
            try:
                action = int(algo.compute_single_action(agent_obs, policy_id="shared_policy"))
            except Exception:
                action = 0
            actions[agent_id] = action

        # Snapshot cursor positions before the step for placement detection.
        pre_cursors: list[tuple[int, int, int]] = []
        for agent_id in env.possible_agents:
            cp = obs.get(agent_id, {}).get("cursor_pos")
            if cp is not None:
                pre_cursors.append((
                    int(round(cp[0] * 31)) + 1,
                    int(round(cp[1] * 31)) + 1,
                    int(round(cp[2] * 31)) + 1,
                ))
            else:
                pre_cursors.append((1, 1, 1))

        obs_next, rewards, terms, truncs, _ = env.step(actions)
        done = all(terms.values())

        # Build per-agent cursor and placed lists.
        cursors: list[tuple[int, int, int]] = []
        placed: list[bool] = []
        acts: list[int] = []
        rews: list[float] = []

        for i, agent_id in enumerate(env.possible_agents):
            a = actions.get(agent_id, 0)
            r = rewards.get(agent_id, 0.0)
            acts.append(a)
            rews.append(float(r))
            total_rewards[i] += float(r)

            cur = (1, 1, 1)
            if agent_id in obs_next:
                cp = obs_next[agent_id].get("cursor_pos")
                if cp is not None:
                    cur = (
                        int(round(cp[0] * 31)) + 1,
                        int(round(cp[1] * 31)) + 1,
                        int(round(cp[2] * 31)) + 1,
                    )
            cursors.append(cur)

            # Detect placement: agent moved to a new cell (trail mode fills destination).
            placed.append(cur != pre_cursors[i])

        steps.append(MultiVoxelStepData(
            step=step_count,
            actions=acts,
            rewards=rews,
            done=done,
            cursors=cursors,
            placed=placed,
        ))

        step_count += 1
        obs = obs_next

    try:
        env.close()
    except Exception:
        pass

    return MultiVoxelEpisodeData(
        agent_count=agent_count,
        max_steps=max_steps,
        steps=steps,
        total_rewards=total_rewards,
        start_positions=start_positions,
        goal_positions=goal_positions,
        init_filled=init_filled,
    )


def _build_multi_payload(
    episode: MultiVoxelEpisodeData,
    iteration: int,
    episode_reward_mean: float,
    experiment_name: str,
    run_id: str,
    *,
    init_filled_file: str,
) -> dict:
    return {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "iteration": iteration,
        "episode_reward_mean": episode_reward_mean,
        "agent_count": episode.agent_count,
        "max_steps": episode.max_steps,
        "obs_mode": "scalar",
        "episode": {
            "total_reward": sum(episode.total_rewards),
            "steps_taken": len(episode.steps),
            "success": len(episode.steps) < episode.max_steps,
            "init_filled_file": init_filled_file,
            "start_positions": [list(p) if p else None for p in episode.start_positions],
            "goal_positions":  [list(p) if p else None for p in episode.goal_positions],
            "steps": [
                {
                    "step":            s.step,
                    "actions":         s.actions,
                    "rewards":         s.rewards,
                    "done":            s.done,
                    "cursors":         [list(c) for c in s.cursors],
                    "placed_per_agent": s.placed,
                }
                for s in episode.steps
            ],
        },
    }


class MultiTrajectoryWriter:
    """Like TrajectoryWriter but for MultiVoxelEpisodeData."""

    def __init__(self, store: "OutputStore", trajectory_every: int, best_trajectory: bool = True) -> None:
        self._store = store
        self._trajectory_every = trajectory_every
        self._best_trajectory = best_trajectory
        self._buffer: list[MultiVoxelEpisodeData] = []
        self._best_reward: float = float("-inf")

    def record(self, episode: MultiVoxelEpisodeData) -> None:
        self._buffer.append(episode)

    def _write_snapshot(
        self,
        json_path: str,
        episode: MultiVoxelEpisodeData,
        iteration: int,
        episode_reward_mean: float,
        experiment_name: str,
        run_id: str,
    ) -> None:
        init_filled_file = _init_filled_sidecar_path(json_path)
        _write_init_filled_sidecar(self._store, init_filled_file, episode.init_filled)
        payload = _build_multi_payload(
            episode,
            iteration,
            episode_reward_mean,
            experiment_name,
            run_id,
            init_filled_file=init_filled_file.rsplit("/", 1)[-1],
        )
        self._store.write_bytes(json_path, json.dumps(payload, indent=2).encode())

    def on_iteration_end(
        self,
        iteration: int,
        episode_reward_mean: float,
        experiment_name: str,
        run_id: str,
        force: bool = False,
    ) -> list[str]:
        if not self._buffer:
            return []
        best_ep = max(self._buffer, key=lambda e: sum(e.total_rewards))
        self._buffer.clear()
        written: list[str] = []

        if (
            force
            or iteration == 1
            or (self._trajectory_every and iteration % self._trajectory_every == 0)
        ):
            path = f"trajectories/iter_{iteration:06d}.json"
            self._write_snapshot(path, best_ep, iteration, episode_reward_mean, experiment_name, run_id)
            written.append(path)

        if self._best_trajectory and episode_reward_mean > self._best_reward:
            self._best_reward = episode_reward_mean
            self._write_snapshot("trajectories/best.json", best_ep, iteration, episode_reward_mean, experiment_name, run_id)
            self._store.write_json(
                "trajectories/best_meta.json",
                {"iteration": iteration, "episode_reward_mean": episode_reward_mean},
            )
            if "trajectories/best.json" not in written:
                written.append("trajectories/best.json")
        return written


def _extract_voxel_count(obs: Any) -> int:
    if not isinstance(obs, dict):
        return 0
    vc = obs.get("voxel_count")
    if vc is None:
        return 0
    try:
        return int(float(vc[0]) if hasattr(vc, "__len__") else float(vc))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# TrajectoryWriter
# ---------------------------------------------------------------------------

class TrajectoryWriter:
    """
    Accumulates VoxelEpisodeData records for one training iteration, then
    serialises the best episode to JSON at configured save points.

    Typical usage (wired into ExperimentRunner):

        writer = TrajectoryWriter(store, trajectory_every=10, best_trajectory=True)

        # In on_iteration_end hook:
        episode = collect_eval_episode(trainer._algo, env_config)
        writer.record(episode)
        written = writer.on_iteration_end(iteration, mean_reward, exp_name, run_id)
    """

    def __init__(
        self,
        store: "OutputStore",
        trajectory_every: int,
        best_trajectory: bool = True,
    ) -> None:
        self._store = store
        self._trajectory_every = trajectory_every
        self._best_trajectory = best_trajectory
        self._buffer: list[VoxelEpisodeData] = []
        self._best_reward: float = float("-inf")

    def record(self, episode: VoxelEpisodeData) -> None:
        """Buffer one episode for the current iteration."""
        self._buffer.append(episode)

    def _write_snapshot(
        self,
        json_path: str,
        episode: VoxelEpisodeData,
        iteration: int,
        episode_reward_mean: float,
        experiment_name: str,
        run_id: str,
    ) -> None:
        init_filled_file = _init_filled_sidecar_path(json_path)
        _write_init_filled_sidecar(self._store, init_filled_file, episode.init_filled)
        payload = _build_payload(
            episode,
            iteration,
            episode_reward_mean,
            experiment_name,
            run_id,
            init_filled_file=init_filled_file.rsplit("/", 1)[-1],
        )
        self._store.write_bytes(json_path, json.dumps(payload, indent=2).encode())

    def on_iteration_end(
        self,
        iteration: int,
        episode_reward_mean: float,
        experiment_name: str,
        run_id: str,
        force: bool = False,
    ) -> list[str]:
        """
        Select the best-rewarded episode from the buffer, clear the buffer,
        and write JSON files according to the save policy.

        Returns a list of relative paths written (may be empty).
        """
        if not self._buffer:
            return []

        best_ep = max(self._buffer, key=lambda e: e.total_reward)
        self._buffer.clear()

        written: list[str] = []

        if (
            force
            or iteration == 1
            or (self._trajectory_every and iteration % self._trajectory_every == 0)
        ):
            path = f"trajectories/iter_{iteration:06d}.json"
            self._write_snapshot(path, best_ep, iteration, episode_reward_mean, experiment_name, run_id)
            written.append(path)

        if self._best_trajectory and episode_reward_mean > self._best_reward:
            self._best_reward = episode_reward_mean
            self._write_snapshot("trajectories/best.json", best_ep, iteration, episode_reward_mean, experiment_name, run_id)
            self._store.write_json(
                "trajectories/best_meta.json",
                {"iteration": iteration, "episode_reward_mean": episode_reward_mean},
            )
            if "trajectories/best.json" not in written:
                written.append("trajectories/best.json")

        return written

    @staticmethod
    def load(store: "OutputStore", rel_path: str) -> dict:
        """Deserialise a trajectory JSON file to a plain dict."""
        return json.loads(store.read_bytes(rel_path).decode())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_payload(
    episode: VoxelEpisodeData,
    iteration: int,
    episode_reward_mean: float,
    experiment_name: str,
    run_id: str,
    *,
    init_filled_file: str,
) -> dict:
    return {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "iteration": iteration,
        "episode_reward_mean": episode_reward_mean,
        "grid_size": episode.grid_size,
        "agent_count": episode.agent_count,
        "max_steps": episode.max_steps,
        "obs_mode": episode.obs_mode,
        "episode": {
            "total_reward": episode.total_reward,
            "steps_taken": len(episode.steps),
            "success": episode.success,
            "init_filled_file": init_filled_file,
            "start_pos": list(episode.start_pos) if episode.start_pos else None,
            "goal_pos": list(episode.goal_pos) if episode.goal_pos else None,
            "steps": [
                {
                    "step": s.step,
                    "action": s.action,
                    "reward": s.reward,
                    "done": s.done,
                    "cursor_x": s.cursor_x,
                    "cursor_y": s.cursor_y,
                    "cursor_z": s.cursor_z,
                    "voxel_count": s.voxel_count,
                    "placed": s.placed,
                }
                for s in episode.steps
            ],
        },
    }


def _init_filled_sidecar_path(json_path: str) -> str:
    return json_path.removesuffix(".json") + "_init_filled.npy"


def _write_init_filled_sidecar(
    store: "OutputStore",
    rel_path: str,
    init_filled: list[tuple[int, int, int]],
) -> None:
    array = np.asarray(init_filled, dtype=np.uint16).reshape((-1, 3))
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    store.write_bytes(rel_path, buffer.getvalue())
