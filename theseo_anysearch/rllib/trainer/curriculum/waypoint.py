"""Waypoint curriculum lifecycle and retention evaluation."""

from __future__ import annotations

from typing import Any

from theseo_anysearch.experiments.trajectory import EpisodeRunMetrics
from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
    WaypointCurriculum,
    WaypointCurriculumState,
    broadcast_waypoint_curriculum,
)


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
        self._apply_state(env_config)
        broadcast_waypoint_curriculum(algorithm, self.curriculum)
        self._persist(store)

    def stage_metric(self) -> dict[str, float]:
        """Return the current stage as a reportable scalar."""
        return {"curriculum/stage": float(self.curriculum.state.stage)}

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
            if isinstance(stage, dict):
                start = stage["start"]
                goal = stage["waypoints"][-1]
                stage_env["waypoint_route"] = stage
                stage_env.pop("waypoints", None)
            else:
                start, goal = stage
                stage_env["waypoints"] = {"start": start, "goal": goal}
                stage_env.pop("waypoint_route", None)
            stage_env["waypoint_curriculum"] = {"enabled": False}
            episodes = collect_rllib_evaluation_episodes(
                algorithm,
                stage_env,
                retention.episodes,
                seed=self.evaluation.seed + stage_index * retention.episodes,
                multi_agent=False,
                num_envs_per_env_runner=self.evaluation.num_envs_per_env_runner,
            )
            metrics = EpisodeRunMetrics.from_voxel_episodes(episodes)
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
                "training_sampling_probabilities": self.curriculum.sampling_probabilities(),
            },
        )
        return {
            "curriculum/stage": float(self.curriculum.state.stage),
            "curriculum/transition": float(transitioned),
            "curriculum/retention_success_rate": overall_rate,
            "curriculum/retention_pass": float(passed),
        }

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
