"""Waypoint curriculum lifecycle and retention evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from theseo_anysearch.experiments.trajectory import EpisodeRunMetrics
from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
    WaypointCurriculum,
    WaypointCurriculumState,
    broadcast_waypoint_curriculum,
)
from theseo_anysearch.rllib.trainer.waypoint_routes import WaypointRoute


def build_route_evaluation_suite(
    curriculum: WaypointCurriculum,
    env_config: dict[str, Any],
    stage_index: int,
    episode_count: int,
    seed_start: int,
) -> list[tuple[int, WaypointRoute]]:
    """Build a stable set of distinct routes without consuming training randomness."""
    routes: list[tuple[int, WaypointRoute]] = []
    signatures: set[tuple[Any, ...]] = set()
    candidate_seed = seed_start
    maximum_attempts = max(episode_count * 100, 1)
    for _ in range(episode_count):
        for _ in range(maximum_attempts):
            route = curriculum.route_for_stage(
                env_config,
                stage_index,
                seed=candidate_seed,
            )
            signature = (route.start, *route.waypoints)
            route_seed = candidate_seed
            candidate_seed += 1
            if signature in signatures:
                continue
            signatures.add(signature)
            routes.append((route_seed, route))
            break
        else:
            raise RuntimeError(
                "could not generate enough distinct curriculum evaluation routes"
            )
    return routes


def _route_summary(seed: int, route: WaypointRoute) -> dict[str, Any]:
    payload = route.model_dump(mode="json")
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "seed": seed,
        "fingerprint": fingerprint,
        "start": route.start,
        "goal": route.goal,
        "waypoint_count": len(route.waypoints),
    }


class CurriculumController:
    """Coordinate curriculum state, worker broadcasts, and stage retention."""

    def __init__(self, env: Any, evaluation: Any) -> None:
        curriculum_config = env.waypoint_curriculum
        evaluation_advance = evaluation.waypoint_curriculum.advance
        if evaluation_advance is not None:
            curriculum_config = curriculum_config.model_copy(update={"advance": evaluation_advance})
        self.curriculum = WaypointCurriculum(curriculum_config, env.to_runtime_dict())
        self.evaluation = evaluation

    def initialize(self, algorithm: Any, store: Any, env_config: dict[str, Any]) -> None:
        """Restore, apply, persist, and broadcast the active stage."""
        if store.exists("curriculum/state.json"):
            self.curriculum.state = WaypointCurriculumState.model_validate(
                store.read_json("curriculum/state.json")
            )
            self.curriculum.clamp_restored_state()
        self._apply_state(env_config)
        broadcast_waypoint_curriculum(algorithm, self.curriculum)
        self._persist(store)

    def stage_metric(self) -> dict[str, float]:
        """Return the current stage as a reportable scalar."""
        maximum_stage = self.curriculum.maximum_stage
        return {
            "curriculum/stage": float(self.curriculum.state.stage),
            "curriculum/terminal": float(self.curriculum.terminal),
            "curriculum/max_stage": float(
                maximum_stage if maximum_stage is not None else -1
            ),
        }

    def evaluate(
        self,
        algorithm: Any,
        iteration: int,
        env_config: dict[str, Any],
        store: Any,
    ) -> dict[str, float]:
        """Run due retention episodes and advance when all gates pass."""
        retention = self.evaluation.waypoint_curriculum
        if not retention.enabled or iteration % retention.frequency != 0:
            return self.stage_metric()

        from theseo_anysearch.rllib.trainer.evaluation.parallel import (
            collect_rllib_evaluation_episodes,
        )

        stage_results: list[dict[str, Any]] = []
        total_finishes = 0
        total_episodes = 0
        for stage_index, stage in enumerate(self.curriculum.stages()):
            stage_env = dict(env_config)
            stage_seed = self.evaluation.seed + stage_index * 100_000
            if isinstance(stage, dict):
                start = stage["start"]
                goal = stage["waypoints"][-1]
                route_suite = build_route_evaluation_suite(
                    self.curriculum,
                    env_config,
                    stage_index,
                    retention.episodes,
                    stage_seed,
                )
                episodes = []
                route_summaries = []
                for episode_index, (route_seed, route) in enumerate(route_suite):
                    route_env = dict(stage_env)
                    route_env["waypoint_route"] = route.model_dump(mode="python")
                    route_env.pop("waypoints", None)
                    route_env["waypoint_curriculum"] = {"enabled": False}
                    episodes.extend(
                        collect_rllib_evaluation_episodes(
                            algorithm,
                            route_env,
                            1,
                            seed=stage_seed + episode_index,
                            multi_agent=False,
                            num_envs_per_env_runner=(
                                self.evaluation.num_envs_per_env_runner
                            ),
                            sync_weights=episode_index == 0,
                        )
                    )
                    route_summaries.append(_route_summary(route_seed, route))
            else:
                start, goal = stage
                stage_env["waypoints"] = {"start": start, "goal": goal}
                stage_env.pop("waypoint_route", None)
                stage_env["waypoint_curriculum"] = {"enabled": False}
                episodes = collect_rllib_evaluation_episodes(
                    algorithm,
                    stage_env,
                    retention.episodes,
                    seed=stage_seed,
                    multi_agent=False,
                    num_envs_per_env_runner=self.evaluation.num_envs_per_env_runner,
                )
                route_summaries = []
            metrics = EpisodeRunMetrics.from_voxel_episodes(episodes)
            completion_values = [
                float((episode.final_info or {}).get(
                    "route_waypoint_completion_fraction", 0.0
                ))
                for episode in episodes
            ]
            total_finishes += metrics.finish_count
            total_episodes += len(episodes)
            stage_results.append(
                {
                    "stage": stage_index,
                    "start": start,
                    "goal": goal,
                    "episodes": len(episodes),
                    "goals_reached": metrics.finish_count,
                    "success_rate": metrics.finish_rate,
                    "completion_fraction": (
                        sum(completion_values) / len(completion_values)
                        if completion_values else 0.0
                    ),
                    "routes": route_summaries,
                }
            )

        overall_rate = total_finishes / total_episodes
        self.curriculum.record_stage_evaluations(
            [(item["episodes"], item["goals_reached"]) for item in stage_results]
        )
        passed = overall_rate >= retention.min_success_rate and all(
            item["success_rate"] >= retention.min_per_stage_success_rate
            for item in stage_results
        )
        transitioned = False
        if passed:
            transitioned = self.curriculum.observe(
                iteration,
                stage_results[-1]["goals_reached"],
            )
        if transitioned:
            self.curriculum.advance_stage(iteration, self.curriculum.sample_stage(env_config))
            self._apply_state(env_config)
        broadcast_waypoint_curriculum(algorithm, self.curriculum)
        self._persist(store)
        store.write_json(
            f"evaluation/curriculum_iter_{iteration:06d}.json",
            {
                "stages": stage_results,
                "passed": passed,
                "terminal": self.curriculum.terminal,
                "maximum_stage": self.curriculum.maximum_stage,
                "training_sampling_probabilities": self.curriculum.sampling_probabilities(),
            },
        )
        sampling = self.curriculum.sampling_probabilities()
        scalars = {
            **self.stage_metric(),
            "curriculum/retention_success_rate": overall_rate,
        }
        for item, probability in zip(stage_results, sampling):
            prefix = f"curriculum/stage_{item['stage']}"
            scalars[f"{prefix}/success_rate"] = float(item["success_rate"])
            scalars[f"{prefix}/completion_fraction"] = float(
                item["completion_fraction"]
            )
            scalars[f"{prefix}/sampling_probability"] = float(probability)
        return scalars

    def _apply_state(self, env_config: dict[str, Any]) -> None:
        state = self.curriculum.state
        if state.waypoints:
            env_config["waypoint_route"] = {"start": state.start, "waypoints": state.waypoints}
            env_config.pop("waypoints", None)
        else:
            env_config["waypoints"] = {"start": state.start, "goal": state.goal}
            env_config.pop("waypoint_route", None)

    def _persist(self, store: Any) -> None:
        store.write_json("curriculum/state.json", self.curriculum.state.model_dump(mode="json"))
