"""Collect and persist heuristic-labeled observation/action datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict
from ray.rllib.models import ModelCatalog

from theseo_anysearch.environments.action_spaces import shortest_actions
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic import (
    VoxelReplanningAStarHeuristic,
    build_voxel_heuristic,
)
from theseo_anysearch.imitation.models import (
    DemonstrationManifest,
    ImitationConfig,
)


def _configure_waypoint_curriculum(
    env: VoxelEnv,
    env_config: dict[str, Any],
    stage_selection: str,
) -> tuple[Any | None, int]:
    """Return the curriculum and number of stages used for collection."""

    from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
        configure_initial_waypoint_curriculum,
    )

    if stage_selection == "initial":
        configure_initial_waypoint_curriculum(env, env_config)
        return None, 0
    from theseo_anysearch.models import WaypointCurriculumConfig
    from theseo_anysearch.rllib.trainer.waypoint_curriculum import WaypointCurriculum

    raw_curriculum = env_config.get("waypoint_curriculum") or {}
    curriculum_config = WaypointCurriculumConfig.model_validate(raw_curriculum)
    if not curriculum_config.enabled:
        raise ValueError(
            "collection.curriculum_stages: all requires an enabled waypoint curriculum"
        )
    curriculum = WaypointCurriculum(curriculum_config, env_config)
    stage_count = len(curriculum.configured_route_stages(env_config))
    return curriculum, stage_count


def _route_action_plan(
    env: VoxelEnv, env_config: dict[str, Any]
) -> list[int | tuple[int, int, int]] | None:
    """Return a fast empty-grid plan for an active waypoint route."""

    raw_curriculum = env_config.get("waypoint_curriculum") or {}
    if raw_curriculum.get("completion_mode") != "continue_route":
        return None
    raw_goal = env._rust_env.goal_pos()
    if raw_goal is None:
        return None
    points = [
        tuple(int(value) for value in env._rust_env.cursor_pos()),
        tuple(int(value) for value in raw_goal),
        *(tuple(int(value) for value in goal) for goal in env._route_remaining),
    ]
    action_mode = str(env_config.get("action_mode", "discrete_26"))
    actions: list[int | tuple[int, int, int]] = []
    for start, goal in zip(points, points[1:]):
        actions.extend(shortest_actions(start, goal, action_mode))
    return actions


class DemonstrationDataset(BaseModel):
    """Flattened policy observations, teacher actions, and episode membership."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_observations: np.ndarray
    train_actions: np.ndarray
    validation_observations: np.ndarray
    validation_actions: np.ndarray
    manifest: DemonstrationManifest


def dataset_fingerprint(
    env_config: dict[str, Any],
    imitation: ImitationConfig,
    observation_size: int,
    action_count: int | list[int],
) -> str:
    """Hash every contract that affects demonstration compatibility."""

    normalized_env = dict(env_config)
    # Each run receives an identical copied native extension under a unique
    # run directory. Its absolute manifest path is runtime plumbing, not part
    # of the observation, action, geometry, or teacher contract.
    normalized_env.pop("native_extension_manifest", None)
    # Demonstration resets always use collection.seed_start + attempt, so the
    # Tune trial's rollout seed offset cannot affect collected examples.
    normalized_env.pop("seed", None)
    for path_key in ("stl_path", "waypoints_file"):
        path_value = normalized_env.get(path_key)
        if path_value:
            normalized_env[path_key] = str(Path(str(path_value)).resolve())
    geometry_pool = normalized_env.get("geometry_pool")
    if isinstance(geometry_pool, dict) and geometry_pool.get("pool_dir"):
        normalized_env["geometry_pool"] = {
            **geometry_pool,
            "pool_dir": str(Path(str(geometry_pool["pool_dir"])).resolve()),
        }

    payload = {
        # Version 3 makes all-stage datasets use one newly sampled route per
        # episode rather than repeating one fixed route per stage.
        "schema_version": 3,
        "env": normalized_env,
        "geometry": _geometry_fingerprint(env_config),
        "teacher": imitation.teacher.model_dump(mode="json"),
        "collection": {
            "episodes": imitation.collection.episodes,
            "seed_start": imitation.collection.seed_start,
            "max_attempts": imitation.collection.max_attempts,
            "require_success": imitation.collection.require_success,
            "validation_fraction": imitation.collection.validation_fraction,
            "curriculum_stages": imitation.collection.curriculum_stages,
        },
        "observation_size": observation_size,
        "action_spec": action_count,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _geometry_fingerprint(env_config: dict[str, Any]) -> str:
    """Hash configured geometry contents rather than only their path names."""

    digest = hashlib.sha256()
    stl_path = env_config.get("stl_path")
    if stl_path:
        path = Path(str(stl_path))
        if path.is_file():
            digest.update(path.read_bytes())
    pool = env_config.get("geometry_pool") or {}
    pool_dir_value = pool.get("pool_dir") if isinstance(pool, dict) else None
    if pool_dir_value:
        pool_dir = Path(str(pool_dir_value))
        if pool_dir.is_dir():
            for path in sorted(pool_dir.rglob("*.npy")):
                digest.update(str(path.relative_to(pool_dir)).encode("utf-8"))
                digest.update(path.read_bytes())
    digest.update(
        json.dumps(env_config.get("geometry_boxes") or [], sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()

def collect_demonstrations(
    env_config: dict[str, Any],
    imitation: ImitationConfig,
) -> DemonstrationDataset:
    """Collect successful teacher rollouts from actual environment transitions."""

    env = VoxelEnv(env_config)
    route_curriculum, curriculum_stage_count = _configure_waypoint_curriculum(
        env, env_config, imitation.collection.curriculum_stages
    )
    preprocessor = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    observation_size = int(np.prod(preprocessor.shape))
    action_nvec = (
        [int(value) for value in env.action_space.nvec]
        if hasattr(env.action_space, "nvec")
        else None
    )
    action_count = sum(action_nvec) if action_nvec else int(env.action_space.n)
    observations: list[np.ndarray] = []
    actions: list[int | tuple[int, int, int]] = []
    episode_ids: list[int] = []
    accepted_seeds: list[int] = []
    teacher_successes = 0
    attempts = 0
    used_routes: set[tuple[Any, ...]] = set()
    stage_episode_counts = [0] * curriculum_stage_count
    if curriculum_stage_count:
        episodes_per_stage, extra_episodes = divmod(
            imitation.collection.episodes,
            curriculum_stage_count,
        )
        stage_targets = [
            episodes_per_stage + int(index < extra_episodes)
            for index in range(curriculum_stage_count)
        ]
    else:
        stage_targets = []
    stage_index = 0

    while (
        len(accepted_seeds) < imitation.collection.episodes
        and attempts < imitation.collection.max_attempts
    ):
        seed = imitation.collection.seed_start + attempts
        attempts += 1
        if route_curriculum is not None:
            route_seed = seed
            for _ in range(imitation.collection.max_attempts):
                route = route_curriculum.route_for_stage(
                    env_config,
                    stage_index,
                    seed=route_seed,
                )
                signature = (route.start, *route.waypoints)
                if signature not in used_routes:
                    used_routes.add(signature)
                    break
                route_seed += imitation.collection.max_attempts
            else:
                raise RuntimeError(
                    "could not generate a unique waypoint route for demonstration collection"
                )
            env.set_waypoint_curriculum(
                [route.model_dump(mode="python")],
                [1.0],
            )
        observation, _ = env.reset(seed=seed)
        episode_observations: list[np.ndarray] = []
        episode_actions: list[int | tuple[int, int, int]] = []
        success = False

        try:
            route_plan = _route_action_plan(env, env_config)
            if route_plan is not None:
                action_plan = route_plan
            else:
                teacher = build_voxel_heuristic(
                    env,
                    imitation.teacher.type,
                    weight=imitation.teacher.weight,
                )
                if isinstance(teacher, VoxelReplanningAStarHeuristic):
                    action_plan = None
                else:
                    action_plan = list(teacher.plan().action_indices)

            step_index = 0
            while True:
                if action_plan is None:
                    current_plan = teacher.plan()
                    if not current_plan.action_indices:
                        break
                    action = current_plan.action_indices[0]
                else:
                    if step_index >= len(action_plan):
                        break
                    action = action_plan[step_index]

                episode_observations.append(
                    np.asarray(preprocessor.transform(observation), dtype=np.float32)
                )
                episode_actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
                step_index += 1
                success = bool(info.get("goal_reached", False))
                if terminated or truncated:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            success = False

        if success or not imitation.collection.require_success:
            episode_id = len(accepted_seeds)
            observations.extend(episode_observations)
            actions.extend(episode_actions)
            episode_ids.extend([episode_id] * len(episode_actions))
            accepted_seeds.append(seed)
            teacher_successes += int(success)
            if route_curriculum is not None:
                stage_episode_counts[stage_index] += 1
                for offset in range(1, curriculum_stage_count + 1):
                    candidate = (stage_index + offset) % curriculum_stage_count
                    if stage_episode_counts[candidate] < stage_targets[candidate]:
                        stage_index = candidate
                        break

    env.close()
    if len(accepted_seeds) < imitation.collection.episodes:
        raise RuntimeError(
            "Heuristic demonstration collection produced "
            f"{len(accepted_seeds)}/{imitation.collection.episodes} episodes "
            f"after {attempts} attempts"
        )
    if not observations:
        raise RuntimeError("Heuristic demonstration collection produced no samples")

    rng = np.random.default_rng(imitation.collection.seed_start)
    unique_episodes = np.arange(len(accepted_seeds), dtype=np.int64)
    rng.shuffle(unique_episodes)
    validation_count = max(
        1,
        int(round(len(unique_episodes) * imitation.collection.validation_fraction)),
    )
    validation_count = min(validation_count, len(unique_episodes) - 1)
    validation_episodes = set(unique_episodes[:validation_count].tolist())
    episode_array = np.asarray(episode_ids, dtype=np.int64)
    validation_mask = np.asarray(
        [episode_id in validation_episodes for episode_id in episode_array],
        dtype=bool,
    )
    observation_array = np.stack(observations).astype(np.float32, copy=False)
    action_array = np.asarray(actions, dtype=np.int64)
    fingerprint = dataset_fingerprint(
        env_config,
        imitation,
        observation_size,
        action_nvec or action_count,
    )
    manifest = DemonstrationManifest(
        fingerprint=fingerprint,
        teacher_type=imitation.teacher.type,
        teacher_weight=imitation.teacher.weight,
        requested_episodes=imitation.collection.episodes,
        successful_episodes=teacher_successes,
        accepted_episodes=len(accepted_seeds),
        attempted_episodes=attempts,
        training_episodes=len(unique_episodes) - validation_count,
        validation_episodes=validation_count,
        training_samples=int((~validation_mask).sum()),
        validation_samples=int(validation_mask.sum()),
        observation_size=observation_size,
        action_count=action_count,
        action_nvec=action_nvec,
        seeds=accepted_seeds,
        stage_episode_counts=(
            stage_episode_counts if route_curriculum is not None else None
        ),
    )
    return DemonstrationDataset(
        train_observations=observation_array[~validation_mask],
        train_actions=action_array[~validation_mask],
        validation_observations=observation_array[validation_mask],
        validation_actions=action_array[validation_mask],
        manifest=manifest,
    )


def save_dataset(dataset: DemonstrationDataset, directory: Path) -> None:
    """Write a compressed dataset and human-readable manifest."""

    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory.joinpath("demonstrations.npz"),
        train_observations=dataset.train_observations,
        train_actions=dataset.train_actions,
        validation_observations=dataset.validation_observations,
        validation_actions=dataset.validation_actions,
    )
    directory.joinpath("manifest.json").write_text(
        dataset.manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_compatible_dataset(
    directory: Path,
    expected_fingerprint: str,
) -> DemonstrationDataset:
    """Load a dataset only when its complete contract fingerprint matches."""

    manifest = DemonstrationManifest.model_validate_json(
        directory.joinpath("manifest.json").read_text(encoding="utf-8")
    )
    if manifest.fingerprint != expected_fingerprint:
        raise ValueError(
            "Imitation dataset fingerprint mismatch: "
            f"expected {expected_fingerprint}, found {manifest.fingerprint}"
        )
    arrays = np.load(directory.joinpath("demonstrations.npz"))
    return DemonstrationDataset(
        train_observations=arrays["train_observations"],
        train_actions=arrays["train_actions"],
        validation_observations=arrays["validation_observations"],
        validation_actions=arrays["validation_actions"],
        manifest=manifest,
    )
