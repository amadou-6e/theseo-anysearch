"""Collect, serialize, and manage evaluation trajectories for runs and sweeps."""

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from theseo_anysearch.worlds.extent import resolve_extent

if TYPE_CHECKING:
    from theseo_anysearch.experiments.output import OutputStore


TRAJECTORY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class WorldArtifactReference:
    """Portable reference to one immutable compiled base world."""

    identity_sha256: str
    schema_version: int
    coordinate_type: str
    extent: tuple[int, int, int]
    manifest_path: str
    source_root: Path


@dataclass(frozen=True)
class VoxelMutationData:
    """Final mutation of one base-world coordinate after a trajectory step."""

    coordinate: tuple[int, int, int]
    occupied: bool
    kind: int = 0
    active: bool = False
    reward_weight: float = 0.0


# ---------------------------------------------------------------------------
# Data classes — filled in by collect_eval_episode(); also usable directly
# ---------------------------------------------------------------------------

@dataclass
class VoxelStepData:
    """Per-step trajectory record for a single-agent voxel episode.

    Parameters
    ----------
    step : int
        Zero-based step index.
    action : int
        Discrete action index selected by the policy.
    reward : float
        Reward received after the action.
    done : bool
        Whether the episode ended after this step.
    cursor_x, cursor_y, cursor_z : int
        Cursor position after the step.
    voxel_count : int
        Total filled voxel count after the step.
    placed : bool
        Whether this step created a new filled voxel.
    """
    step: int
    action: int        # 0..25 = index into 26-neighbor cube {-1,0,1}³ \ origin
    reward: float
    done: bool
    cursor_x: int      # cursor position AFTER the step (destination for movement)
    cursor_y: int
    cursor_z: int
    voxel_count: int   # total filled voxels AFTER this step
    placed: bool       # True if a new voxel was filled this step (Place or trail)
    reward_breakdown: dict[str, float] | None = None
    termination_reason: str = "in_progress"
    mutations: list[VoxelMutationData] | None = None


@dataclass
class VoxelEpisodeData:
    """Trajectory record for one single-agent evaluation episode.

    Parameters
    ----------
    agent_count : int
        Number of agents represented in the episode.
    max_steps : int
        Episode step limit.
    obs_mode : str
        Observation mode used during evaluation.
    init_filled : list[tuple[int, int, int]]
        Filled voxels present at episode start.
    steps : list[VoxelStepData]
        Recorded per-step data.
    total_reward : float
        Total episode reward.
    success : bool
        Whether the goal was reached before timeout.
    grid_size : int
        Side length of the voxel grid.
    start_pos : tuple[int, int, int] | None
        Start position, when available.
    goal_pos : tuple[int, int, int] | None
        Goal position, when available.
    """
    agent_count: int
    max_steps: int
    obs_mode: str
    init_filled: list[tuple[int, int, int]]  # filled voxels at episode start
    steps: list[VoxelStepData]
    total_reward: float
    success: bool      # True if episode ended before max_steps (reached target)
    grid_size: int = 32
    extent: tuple[int, int, int] = (32, 32, 32)
    start_pos: tuple[int, int, int] | None = None
    goal_pos: tuple[int, int, int] | None = None
    termination_reason: str = "unknown"
    initial_goal_distance: float | None = None
    final_goal_distance: float | None = None
    minimum_goal_distance: float | None = None
    reward_breakdown: dict[str, float] | None = None
    unshaped_return: float | None = None
    final_info: dict[str, Any] | None = None
    world: WorldArtifactReference | None = None


@dataclass
class EpisodeRunMetrics:
    """TensorBoard-friendly metrics derived from one evaluation trajectory.

    Parameters
    ----------
    collision_count : int
        Number of blocked movement steps inferred from the trajectory.
    collision_rate : float
        Collision count divided by steps taken.
    finish_count : int
        Number of successful finishes represented by the episode summary.
    finish_rate : float
        Finish count divided by the number of summarized episodes.
    mean_steps_on_success : float
        Steps taken when the episode succeeds, else ``0.0``.
    goal_progress_mean : float
        Net Manhattan-distance reduction from start to final cursor position.
    """

    collision_count: int
    collision_rate: float
    finish_count: int
    finish_rate: float
    mean_steps_on_success: float
    goal_progress_mean: float

    @classmethod
    def from_voxel_episode(cls, episode: VoxelEpisodeData) -> "EpisodeRunMetrics":
        """Summarize one single-agent evaluation trajectory."""
        steps_taken = len(episode.steps)
        finish_count = 1 if episode.success else 0
        collision_count = _count_voxel_collisions(episode)
        return cls(
            collision_count=collision_count,
            collision_rate=(collision_count / steps_taken) if steps_taken else 0.0,
            finish_count=finish_count,
            finish_rate=float(finish_count),
            mean_steps_on_success=float(steps_taken) if episode.success else 0.0,
            goal_progress_mean=_goal_progress(episode.start_pos, _last_cursor(episode), episode.goal_pos),
        )

    @classmethod
    def from_voxel_episodes(
        cls,
        episodes: list[VoxelEpisodeData],
    ) -> "EpisodeRunMetrics":
        """Summarize one deterministic single-agent evaluation batch."""
        if not episodes:
            return cls(0, 0.0, 0, 0.0, 0.0, 0.0)
        per_episode = [cls.from_voxel_episode(episode) for episode in episodes]
        total_steps = sum(len(episode.steps) for episode in episodes)
        collision_count = sum(metrics.collision_count for metrics in per_episode)
        successful_steps = [
            len(episode.steps) for episode in episodes if episode.success
        ]
        return cls(
            collision_count=collision_count,
            collision_rate=(collision_count / total_steps) if total_steps else 0.0,
            finish_count=sum(metrics.finish_count for metrics in per_episode),
            finish_rate=sum(metrics.finish_rate for metrics in per_episode) / len(per_episode),
            mean_steps_on_success=(
                sum(successful_steps) / len(successful_steps)
                if successful_steps
                else 0.0
            ),
            goal_progress_mean=sum(
                metrics.goal_progress_mean for metrics in per_episode
            ) / len(per_episode),
        )

    @classmethod
    def from_multi_voxel_episode(
        cls,
        episode: MultiVoxelEpisodeData,
    ) -> "EpisodeRunMetrics":
        """Summarize one multi-agent evaluation trajectory."""
        steps_taken = len(episode.steps)
        collision_count = _count_multi_voxel_collisions(episode)
        finish_count = sum(
            1
            for start_pos, goal_pos, final_cursor in zip(
                episode.start_positions,
                episode.goal_positions,
                _last_multi_cursors(episode),
            )
            if goal_pos is not None and final_cursor == goal_pos
        )
        goal_progress = [
            _goal_progress(start_pos, final_cursor, goal_pos)
            for start_pos, goal_pos, final_cursor in zip(
                episode.start_positions,
                episode.goal_positions,
                _last_multi_cursors(episode),
            )
        ]
        return cls(
            collision_count=collision_count,
            collision_rate=(collision_count / (steps_taken * episode.agent_count))
            if steps_taken and episode.agent_count
            else 0.0,
            finish_count=finish_count,
            finish_rate=(finish_count / episode.agent_count) if episode.agent_count else 0.0,
            mean_steps_on_success=float(steps_taken) if finish_count else 0.0,
            goal_progress_mean=(sum(goal_progress) / len(goal_progress)) if goal_progress else 0.0,
        )

    @classmethod
    def from_multi_voxel_episodes(
        cls,
        episodes: list[MultiVoxelEpisodeData],
    ) -> "EpisodeRunMetrics":
        """Summarize one deterministic multi-agent evaluation batch."""
        if not episodes:
            return cls(0, 0.0, 0, 0.0, 0.0, 0.0)
        per_episode = [cls.from_multi_voxel_episode(episode) for episode in episodes]
        agent_episodes = sum(episode.agent_count for episode in episodes)
        action_count = sum(
            len(episode.steps) * episode.agent_count for episode in episodes
        )
        finish_count = sum(metrics.finish_count for metrics in per_episode)
        successful_steps = [
            len(episode.steps)
            for episode, metrics in zip(episodes, per_episode)
            for _ in range(metrics.finish_count)
        ]
        return cls(
            collision_count=sum(metrics.collision_count for metrics in per_episode),
            collision_rate=(
                sum(metrics.collision_count for metrics in per_episode) / action_count
                if action_count
                else 0.0
            ),
            finish_count=finish_count,
            finish_rate=(finish_count / agent_episodes) if agent_episodes else 0.0,
            mean_steps_on_success=(
                sum(successful_steps) / len(successful_steps)
                if successful_steps
                else 0.0
            ),
            goal_progress_mean=(
                sum(
                    metrics.goal_progress_mean * episode.agent_count
                    for metrics, episode in zip(per_episode, episodes)
                ) / agent_episodes
                if agent_episodes
                else 0.0
            ),
        )
    def as_scalar_dict(self) -> dict[str, float]:
        """Return non-redundant task diagnostics for TensorBoard."""
        return {
            "eval/task/collision_rate": self.collision_rate,
        }


# ---------------------------------------------------------------------------
# Eval episode collector
# ---------------------------------------------------------------------------

def _compiled_world_reference(
    env_config: dict[str, Any],
) -> WorldArtifactReference | None:
    raw_path = env_config.get("compiled_world_path")
    if raw_path is None:
        return None
    from theseo_anysearch.worlds.compiler import validate_compiled_world

    compiled = validate_compiled_world(Path(raw_path).resolve())
    manifest = compiled.manifest
    identity = manifest.identity_sha256
    return WorldArtifactReference(
        identity_sha256=identity,
        schema_version=manifest.schema_version,
        coordinate_type=manifest.coordinate_type,
        extent=manifest.extent.as_tuple(),
        manifest_path=f"../worlds/{identity}/manifest.json",
        source_root=compiled.root,
    )


def _overlay_snapshot(env: Any) -> dict[tuple[int, int, int], VoxelMutationData]:
    rust_env = getattr(env, "_rust_env", None)
    if rust_env is None or not hasattr(rust_env, "overlay_mutations"):
        return {}
    mutations: dict[tuple[int, int, int], VoxelMutationData] = {}
    for x, y, z, occupied, kind, active, reward_weight in rust_env.overlay_mutations():
        coordinate = (int(x), int(y), int(z))
        mutations[coordinate] = VoxelMutationData(
            coordinate=coordinate,
            occupied=bool(occupied),
            kind=int(kind),
            active=bool(active),
            reward_weight=float(reward_weight),
        )
    return mutations


def _overlay_delta(
    previous: dict[tuple[int, int, int], VoxelMutationData],
    current: dict[tuple[int, int, int], VoxelMutationData],
) -> list[VoxelMutationData]:
    changed = [
        mutation
        for coordinate, mutation in current.items()
        if previous.get(coordinate) != mutation
    ]
    # An overlay-only placement can be removed entirely when no immutable base
    # exists underneath it. Encode that disappearance as an empty resolution.
    changed.extend(
        VoxelMutationData(coordinate=coordinate, occupied=False)
        for coordinate in previous.keys() - current.keys()
    )
    return sorted(changed, key=lambda mutation: mutation.coordinate)

@dataclass
class _VoxelEpisodeState:
    env_config: dict[str, Any]
    env: Any
    obs: Any
    init_filled: list[tuple[int, int, int]]
    start_pos: tuple[int, int, int] | None
    goal_pos: tuple[int, int, int] | None
    prev_voxel_count: int
    steps: list[VoxelStepData]
    world: WorldArtifactReference | None
    overlay: dict[tuple[int, int, int], VoxelMutationData]
    total_reward: float = 0.0
    final_info: dict[str, Any] | None = None
    done: bool = False

    @classmethod
    def create(
        cls,
        env_config: dict[str, Any],
        *,
        env: Any = None,
        seed: int | None = None,
    ) -> "_VoxelEpisodeState":
        if env is None:
            from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

            env = VoxelEnv(env_config)
        from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
            configure_initial_waypoint_curriculum,
        )

        configure_initial_waypoint_curriculum(env, env_config)
        obs, _ = env.reset(seed=seed)
        world = _compiled_world_reference(env_config)
        init_filled: list[tuple[int, int, int]] = []
        start_pos: tuple[int, int, int] | None = None
        goal_pos: tuple[int, int, int] | None = None
        if hasattr(env, "_rust_env") and env._rust_env is not None:
            if world is None:
                init_filled = [
                    (int(x), int(y), int(z))
                    for x, y, z in env._rust_env.filled_voxels()
                ]
            raw_goal = env._rust_env.goal_pos()
            if raw_goal is not None:
                goal_pos = (
                    int(raw_goal[0]),
                    int(raw_goal[1]),
                    int(raw_goal[2]),
                )
            raw_cursor = env._rust_env.cursor_pos()
            start_pos = (
                int(raw_cursor[0]),
                int(raw_cursor[1]),
                int(raw_cursor[2]),
            )
        return cls(
            env_config=env_config,
            env=env,
            obs=obs,
            init_filled=init_filled,
            start_pos=start_pos,
            goal_pos=goal_pos,
            prev_voxel_count=_environment_voxel_count(env, len(init_filled)),
            steps=[],
            world=world,
            overlay=_overlay_snapshot(env),
        )

    def advance(self, raw_action: Any) -> None:
        raw_action = _unwrap_policy_action(raw_action)
        action = (
            int(self.env._encode_action(raw_action))
            if hasattr(self.env, "_encode_action")
            else int(raw_action)
        )
        obs_next, reward, terminated, truncated, info = self.env.step(raw_action)
        self.done = bool(terminated or truncated)
        voxel_count = _environment_voxel_count(self.env, len(self.init_filled))
        cursor = (1, 1, 1)
        if hasattr(self.env, "_rust_env") and self.env._rust_env is not None:
            cursor = self.env._rust_env.cursor_pos()
        overlay = _overlay_snapshot(self.env)
        mutations = _overlay_delta(self.overlay, overlay)
        self.steps.append(VoxelStepData(
            step=len(self.steps),
            action=action,
            reward=float(reward),
            done=self.done,
            cursor_x=int(cursor[0]),
            cursor_y=int(cursor[1]),
            cursor_z=int(cursor[2]),
            voxel_count=voxel_count,
            placed=voxel_count > self.prev_voxel_count,
            reward_breakdown=dict(info.get("reward_breakdown", {})),
            termination_reason=str(info.get("termination_reason", "in_progress")),
            mutations=mutations,
        ))
        self.total_reward += float(reward)
        self.prev_voxel_count = voxel_count
        self.obs = obs_next
        self.final_info = info
        self.overlay = overlay

    def finish(self) -> VoxelEpisodeData:
        self.close()
        final_info = self.final_info or {}
        return VoxelEpisodeData(
            agent_count=1,
            max_steps=self.env_config.get("max_steps", 200),
            obs_mode=self.env_config.get("obs_mode", "scalar"),
            grid_size=max(resolve_extent(self.env_config)),
            extent=resolve_extent(self.env_config),
            init_filled=self.init_filled,
            steps=self.steps,
            total_reward=self.total_reward,
            success=bool(final_info.get("goal_reached", False)),
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            termination_reason=str(final_info.get("termination_reason", "unknown")),
            initial_goal_distance=final_info.get("initial_goal_distance"),
            final_goal_distance=final_info.get("final_goal_distance"),
            minimum_goal_distance=final_info.get("minimum_goal_distance"),
            reward_breakdown=dict(final_info.get("episode_reward_breakdown", {})),
            unshaped_return=sum(
                step.reward
                - (step.reward_breakdown or {}).get("distance_progress", 0.0)
                for step in self.steps
            ),
            final_info=dict(final_info),
            world=self.world,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def _unwrap_policy_action(raw_action: Any) -> Any:
    if (
        isinstance(raw_action, tuple)
        and len(raw_action) == 3
        and isinstance(raw_action[2], dict)
    ):
        return raw_action[0]
    return raw_action


def _single_agent_policy_adapter(algo: Any, observation_space: Any | None = None) -> Any:
    """Use RLModule inference when the algorithm runs on RLlib's modern stack."""

    env_runner = getattr(algo, "env_runner", None)
    if env_runner is None:
        return algo
    module = getattr(env_runner, "module", None)
    if module is None and hasattr(env_runner, "get_module"):
        module = env_runner.get_module()
    if module is None:
        return algo
    from theseo_anysearch.rllib.trainer.evaluation.parallel import _PolicyAdapter

    return _PolicyAdapter(env_runner, observation_space=observation_space)


def collect_eval_episode(
    algo: Any,
    env_config: dict,
    *,
    env: Any = None,
    seed: int | None = None,
) -> VoxelEpisodeData:
    """Run one deterministic single-agent evaluation episode."""
    state = _VoxelEpisodeState.create(env_config, env=env, seed=seed)
    policy = _single_agent_policy_adapter(
        algo,
        getattr(state.env, "observation_space", None),
    )
    try:
        while not state.done:
            action = policy.compute_single_action(
                state.obs,
                policy_id="default_policy",
                explore=False,
            )
            state.advance(action)
        return state.finish()
    finally:
        state.close()


def collect_vectorized_eval_episodes(
    algo: Any,
    env_config: dict,
    seeds: tuple[int, ...],
) -> list[VoxelEpisodeData]:
    """Collect independent episodes with batched policy inference."""
    if not seeds:
        raise ValueError("evaluation episode seeds must not be empty")
    states = [
        _VoxelEpisodeState.create(env_config, seed=seed)
        for seed in seeds
    ]
    active = list(enumerate(states))
    completed: dict[int, VoxelEpisodeData] = {}
    try:
        while active:
            raw_actions = algo.compute_actions(
                [state.obs for _, state in active],
                policy_id="default_policy",
                explore=False,
            )
            if isinstance(raw_actions, tuple):
                raw_actions = raw_actions[0]
            if len(raw_actions) != len(active):
                raise RuntimeError(
                    "policy returned "
                    f"{len(raw_actions)} actions for {len(active)} evaluation environments"
                )
            remaining: list[tuple[int, _VoxelEpisodeState]] = []
            for (episode_index, state), action in zip(active, raw_actions):
                state.advance(action)
                if state.done:
                    completed[episode_index] = state.finish()
                else:
                    remaining.append((episode_index, state))
            active = remaining
        return [completed[index] for index in range(len(states))]
    finally:
        for state in states:
            state.close()


def collect_eval_episodes(
    algo: Any,
    env_config: dict,
    count: int,
    *,
    seed: int | None = None,
) -> list[VoxelEpisodeData]:
    """Collect a deterministic evaluation batch from fresh environments."""
    if count < 1:
        raise ValueError("evaluation episode count must be at least one")
    return [
        collect_eval_episode(
            algo,
            env_config,
            seed=None if seed is None else seed + episode_index,
        )
        for episode_index in range(count)
    ]


def collect_heuristic_episode(
    env_config: dict,
    heuristic_type: str,
    *,
    weight: float | None = None,
    env: Any = None,
    seed: int | None = None,
) -> VoxelEpisodeData:
    """Collect one reference trajectory from a configured voxel heuristic."""

    if env is None:
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

        env = VoxelEnv(env_config)
    env.reset(seed=seed)

    from theseo_anysearch.heuristic import (
        VoxelReplanningAStarHeuristic,
        build_voxel_heuristic,
    )

    rust_env = env._rust_env
    world = _compiled_world_reference(env_config)
    init_filled = [] if world is not None else [
        (int(x), int(y), int(z))
        for x, y, z in rust_env.filled_voxels()
    ]
    raw_start = rust_env.cursor_pos()
    start_pos = (int(raw_start[0]), int(raw_start[1]), int(raw_start[2]))
    raw_goal = rust_env.goal_pos()
    goal_pos = None if raw_goal is None else (
        int(raw_goal[0]),
        int(raw_goal[1]),
        int(raw_goal[2]),
    )

    heuristic = build_voxel_heuristic(
        env,
        heuristic_type,
        weight=weight,
    )
    if isinstance(heuristic, VoxelReplanningAStarHeuristic):
        replay = heuristic.replay()
    else:
        replay = heuristic.replay(heuristic.plan())

    trail_mode = bool(env_config.get("trail_mode", False))
    filled = set(init_filled)
    steps: list[VoxelStepData] = []
    for index, (action, reward, cursor) in enumerate(
        zip(
            replay.action_indices,
            replay.rewards,
            replay.positions[1:],
        )
    ):
        placed = trail_mode and cursor not in filled
        if placed:
            filled.add(cursor)
        steps.append(
            VoxelStepData(
                step=index,
                action=action,
                reward=reward,
                done=(
                    index == replay.steps_executed - 1
                    and (replay.terminated or replay.truncated)
                ),
                cursor_x=cursor[0],
                cursor_y=cursor[1],
                cursor_z=cursor[2],
                voxel_count=len(filled),
                placed=placed,
            )
        )

    try:
        env.close()
    except Exception:
        pass

    return VoxelEpisodeData(
        agent_count=1,
        max_steps=int(env_config.get("max_steps", 200)),
        obs_mode=str(env_config.get("obs_mode", "scalar")),
        grid_size=max(resolve_extent(env_config)),
        extent=resolve_extent(env_config),
        init_filled=init_filled,
        steps=steps,
        total_reward=sum(replay.rewards),
        success=replay.goal_reached,
        start_pos=start_pos,
        goal_pos=goal_pos,
        world=world,
    )


def write_heuristic_trajectory(
    store: "OutputStore",
    episode: VoxelEpisodeData,
    *,
    heuristic_type: str,
    weight: float | None,
    iteration: int,
    experiment_name: str,
    run_id: str,
) -> str:
    """Write a replayer-compatible heuristic reference trajectory."""

    json_path = f"trajectories/heuristic_{heuristic_type}.json"
    init_filled_file = _prepare_geometry_artifact(store, json_path, episode)
    payload = _build_payload(
        episode,
        iteration,
        episode.total_reward,
        experiment_name,
        run_id,
        init_filled_file=(
            init_filled_file.rsplit("/", 1)[-1]
            if init_filled_file is not None
            else None
        ),
    )
    payload["heuristic"] = {"type": heuristic_type, "weight": weight}
    store.write_bytes(json_path, json.dumps(payload, indent=2).encode())
    return json_path

# ---------------------------------------------------------------------------
# Multi-agent data classes
# ---------------------------------------------------------------------------

@dataclass
class MultiVoxelStepData:
    """Per-step trajectory record for a multi-agent voxel episode.

    Parameters
    ----------
    step : int
        Zero-based step index.
    actions : list[int]
        Per-agent action indices.
    rewards : list[float]
        Per-agent rewards for the step.
    done : bool
        Whether the multi-agent episode ended after this step.
    cursors : list[tuple[int, int, int]]
        Per-agent cursor positions after the step.
    placed : list[bool]
        Per-agent placement flags for the step.
    """
    step: int
    actions: list[int]           # per-agent, len == agent_count
    rewards: list[float]         # per-agent
    done: bool
    cursors: list[tuple[int, int, int]]   # per-agent cursor AFTER step
    placed: list[bool]           # per-agent: True if a new voxel was filled
    mutations: list[VoxelMutationData] | None = None


@dataclass
class MultiVoxelEpisodeData:
    """Trajectory record for one multi-agent evaluation episode.

    Parameters
    ----------
    agent_count : int
        Number of agents represented in the episode.
    max_steps : int
        Episode step limit.
    steps : list[MultiVoxelStepData]
        Recorded per-step data.
    total_rewards : list[float]
        Per-agent cumulative rewards.
    start_positions : list[tuple[int, int, int] | None]
        Per-agent start positions.
    goal_positions : list[tuple[int, int, int] | None]
        Per-agent goal positions.
    init_filled : list[tuple[int, int, int]]
        Filled voxels present at episode start.
    """
    agent_count: int
    max_steps: int
    steps: list[MultiVoxelStepData]
    total_rewards: list[float]   # per-agent cumulative
    start_positions: list[tuple[int, int, int] | None]
    goal_positions: list[tuple[int, int, int] | None]
    init_filled: list[tuple[int, int, int]]  # geometry voxels at episode start
    obs_mode: str = "scalar"
    extent: tuple[int, int, int] = (32, 32, 32)
    world: WorldArtifactReference | None = None


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
    world = _compiled_world_reference(env_config)
    agent_count = len(env.possible_agents)

    init_filled: list[tuple[int, int, int]] = []
    start_positions: list[tuple[int, int, int] | None] = [None] * agent_count
    goal_positions:  list[tuple[int, int, int] | None] = [None] * agent_count
    if hasattr(env, "_rust_env") and env._rust_env is not None:
        if world is None:
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
    overlay = _overlay_snapshot(env)

    step_count = 0
    while env.agents:
        actions: dict[str, int] = {}
        for agent_id in env.agents:
            agent_obs = obs[agent_id]
            try:
                action = int(algo.compute_single_action(agent_obs, policy_id="shared_policy"))
            except Exception as exc:
                raise RuntimeError(
                    f"Policy inference failed for agent {agent_id!r} "
                    "using policy 'shared_policy'"
                ) from exc
            actions[agent_id] = action

        # Snapshot cursor positions before the step for placement detection.
        pre_cursors = [
            tuple(int(value) for value in cursor)
            for cursor in env._rust_env.cursor_positions()
        ]

        obs_next, rewards, terms, truncs, _ = env.step(actions)
        done = all(terms.values())

        # Build per-agent cursor and placed lists.
        cursors: list[tuple[int, int, int]] = []
        placed: list[bool] = []
        acts: list[int] = []
        rews: list[float] = []
        native_cursors = env._rust_env.cursor_positions()
        current_overlay = _overlay_snapshot(env)

        for i, agent_id in enumerate(env.possible_agents):
            a = actions.get(agent_id, 0)
            r = rewards.get(agent_id, 0.0)
            acts.append(a)
            rews.append(float(r))
            total_rewards[i] += float(r)

            cur = (
                tuple(int(value) for value in native_cursors[i])
                if i < len(native_cursors)
                else (1, 1, 1)
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
            mutations=_overlay_delta(overlay, current_overlay),
        ))
        overlay = current_overlay

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
        obs_mode=str(env_config.get("obs_mode", "scalar")),
        extent=resolve_extent(env_config),
        world=world,
    )


def _build_multi_payload(
    episode: MultiVoxelEpisodeData,
    iteration: int,
    episode_reward_mean: float,
    experiment_name: str,
    run_id: str,
    *,
    init_filled_file: str | None,
) -> dict:
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "iteration": iteration,
        "episode_reward_mean": episode_reward_mean,
        "agent_count": episode.agent_count,
        "max_steps": episode.max_steps,
        "obs_mode": episode.obs_mode,
        "extent": list(episode.extent),
        "episode": {
            "total_reward": sum(episode.total_rewards),
            "steps_taken": len(episode.steps),
            "success": len(episode.steps) < episode.max_steps,
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
                    "mutations": _mutation_payloads(s.mutations),
                }
                for s in episode.steps
            ],
        },
    }
    _attach_world_or_legacy_geometry(payload, episode.world, init_filled_file)
    return payload


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
        init_filled_file = _prepare_geometry_artifact(self._store, json_path, episode)
        payload = _build_multi_payload(
            episode,
            iteration,
            episode_reward_mean,
            experiment_name,
            run_id,
            init_filled_file=(
                init_filled_file.rsplit("/", 1)[-1]
                if init_filled_file is not None
                else None
            ),
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


def _environment_voxel_count(env: Any, initial_filled_count: int = 0) -> int:
    """Return the current filled-cell count without exposing it to the policy."""
    if hasattr(env, "filled_voxel_count"):
        return int(env.filled_voxel_count())
    rust_env = getattr(env, "_rust_env", None)
    if rust_env is None or not hasattr(rust_env, "filled_voxels"):
        return 0
    return max(len(rust_env.filled_voxels()) - initial_filled_count, 0)


def _last_cursor(episode: VoxelEpisodeData) -> tuple[int, int, int] | None:
    """Return the final cursor position for a single-agent episode."""
    if episode.steps:
        last_step = episode.steps[-1]
        return (last_step.cursor_x, last_step.cursor_y, last_step.cursor_z)
    return episode.start_pos


def _last_multi_cursors(
    episode: MultiVoxelEpisodeData,
) -> list[tuple[int, int, int] | None]:
    """Return the final cursor position for each agent in a multi-agent episode."""
    if episode.steps:
        return list(episode.steps[-1].cursors)
    return list(episode.start_positions)


def _count_voxel_collisions(episode: VoxelEpisodeData) -> int:
    """Infer blocked movement steps from cursor and placement changes."""
    if episode.start_pos is None:
        return 0

    collisions = 0
    prev_cursor = episode.start_pos
    for step in episode.steps:
        cursor = (step.cursor_x, step.cursor_y, step.cursor_z)
        if not step.placed and cursor == prev_cursor:
            collisions += 1
        prev_cursor = cursor
    return collisions


def _count_multi_voxel_collisions(episode: MultiVoxelEpisodeData) -> int:
    """Infer blocked movement steps for all agents in a multi-agent episode."""
    prev_cursors = list(episode.start_positions)
    collisions = 0
    for step in episode.steps:
        for index, cursor in enumerate(step.cursors):
            if not step.placed[index] and prev_cursors[index] == cursor:
                collisions += 1
            prev_cursors[index] = cursor
    return collisions


def _manhattan_distance(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
) -> int:
    """Return Manhattan distance between two voxel coordinates."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1]) + abs(start[2] - goal[2])


def _goal_progress(
    start_pos: tuple[int, int, int] | None,
    end_pos: tuple[int, int, int] | None,
    goal_pos: tuple[int, int, int] | None,
) -> float:
    """Return net Manhattan-distance reduction toward the goal."""
    if start_pos is None or end_pos is None or goal_pos is None:
        return 0.0
    start_distance = _manhattan_distance(start_pos, goal_pos)
    end_distance = _manhattan_distance(end_pos, goal_pos)
    return float(start_distance - end_distance)


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
        init_filled_file = _prepare_geometry_artifact(self._store, json_path, episode)
        payload = _build_payload(
            episode,
            iteration,
            episode_reward_mean,
            experiment_name,
            run_id,
            init_filled_file=(
                init_filled_file.rsplit("/", 1)[-1]
                if init_filled_file is not None
                else None
            ),
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
    init_filled_file: str | None,
) -> dict:
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "iteration": iteration,
        "episode_reward_mean": episode_reward_mean,
        "grid_size": episode.grid_size,
        "extent": list(episode.extent),
        "agent_count": episode.agent_count,
        "max_steps": episode.max_steps,
        "obs_mode": episode.obs_mode,
        "episode": {
            "total_reward": episode.total_reward,
            "steps_taken": len(episode.steps),
            "success": episode.success,
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
                    "mutations": _mutation_payloads(s.mutations),
                }
                for s in episode.steps
            ],
        },
    }
    _attach_world_or_legacy_geometry(payload, episode.world, init_filled_file)
    return payload


def _mutation_payloads(
    mutations: list[VoxelMutationData] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "coordinate": list(mutation.coordinate),
            "occupied": mutation.occupied,
            "kind": mutation.kind,
            "active": mutation.active,
            "reward_weight": mutation.reward_weight,
        }
        for mutation in mutations or []
    ]


def _attach_world_or_legacy_geometry(
    payload: dict[str, Any],
    world: WorldArtifactReference | None,
    init_filled_file: str | None,
) -> None:
    if world is None:
        if init_filled_file is None:
            raise ValueError("legacy trajectory geometry requires an init_filled sidecar")
        payload["episode"]["init_filled_file"] = init_filled_file
        return
    payload["world"] = {
        "identity_sha256": world.identity_sha256,
        "schema_version": world.schema_version,
        "coordinate_type": world.coordinate_type,
        "extent": list(world.extent),
        "manifest_path": world.manifest_path,
    }


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _archive_compiled_world(
    store: "OutputStore",
    world: WorldArtifactReference,
) -> None:
    from theseo_anysearch.worlds.compiler import validate_compiled_world

    destination = store.root.joinpath("worlds", world.identity_sha256)
    if destination.is_dir():
        validate_compiled_world(destination)
        return
    temporary = store.root.joinpath(
        "worlds", f".{world.identity_sha256}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(world.source_root, temporary, copy_function=_link_or_copy)
    validate_compiled_world(temporary)
    try:
        os.replace(temporary, destination)
    except OSError:
        if not destination.is_dir():
            raise
        shutil.rmtree(temporary)
    archived = validate_compiled_world(destination)
    if archived.manifest.identity_sha256 != world.identity_sha256:
        raise ValueError("archived compiled world identity does not match trajectory")


def _prepare_geometry_artifact(
    store: "OutputStore",
    json_path: str,
    episode: VoxelEpisodeData | MultiVoxelEpisodeData,
) -> str | None:
    if episode.world is not None:
        _archive_compiled_world(store, episode.world)
        return None
    init_filled_file = _init_filled_sidecar_path(json_path)
    _write_init_filled_sidecar(store, init_filled_file, episode.init_filled)
    return init_filled_file


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
