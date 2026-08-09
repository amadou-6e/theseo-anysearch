"""Report building and writing for policy explanations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from theseo_anysearch.rllib.explain.features import FeatureSchema
from theseo_anysearch.rllib.explain.models import ExplainedStep, ExplanationReport, ExplanationRequest
from theseo_anysearch.rllib.explain.traces import ObservationTrace, ObservationTraceStep


GEOMETRY_CELL = 1.0 / 3.0
TRAIL_CELL = 2.0 / 3.0
OCCUPIED_RAY_TYPE = 1.0 / 5.0
BOUNDARY_RAY_TYPE = 4.0 / 5.0
FILLED_RAY_TYPE = 5.0 / 5.0
BLOCKING_RAY_TYPES = (OCCUPIED_RAY_TYPE, BOUNDARY_RAY_TYPE, FILLED_RAY_TYPE)


class ExplanationReportBuilder:
    """Build JSON-ready explanation reports from scores and attributions.

    Parameters
    ----------
    schema : FeatureSchema
        Observation feature schema.
    method : str
        Attribution method name.
    """

    def __init__(self, schema: FeatureSchema, method: str) -> None:
        self._schema = schema
        self._method = method

    def build(
        self,
        request: ExplanationRequest,
        trace: ObservationTrace,
        selected_steps: list[int],
        score_rows: Mapping[int, np.ndarray],
        attributions: Mapping[int, dict[str, float]],
        score_type: str,
    ) -> ExplanationReport:
        """Build an explanation report."""

        explained_steps = [
            self._build_step(trace.step(index), score_rows[index], attributions.get(index, {}))
            for index in selected_steps
        ]
        return ExplanationReport(
            run_ref=request.run_ref,
            checkpoint=request.checkpoint,
            trajectory=request.trajectory,
            algorithm=trace.algorithm,
            score_type=score_type,
            method=self._method,
            feature_schema_version=self._schema.version,
            steps=explained_steps,
        )

    def _build_step(
        self,
        step: ObservationTraceStep,
        scores: np.ndarray,
        attributions: dict[str, float],
    ) -> ExplainedStep:
        """Build one explained step from trace metadata and scores."""

        best_safe_action = self.best_safe_action_for_observation(
            step.observation, scores, excluded_action=step.action
        )
        chosen_score = float(scores[step.action])
        best_safe_score = float(scores[best_safe_action])
        local_grid = step.observation.get("local_grid")
        ray_hits = step.observation.get("ray_hits")
        ray_types = step.observation.get("ray_hit_types")
        destination_value = (
            self.destination_value(np.asarray(local_grid), step.action)
            if local_grid is not None else None
        )
        return ExplainedStep(
            step=step.step,
            chosen_action=step.action,
            chosen_direction=self._schema.action_directions[step.action],
            collision_visible=self.collision_visible_observation(
                step.observation, step.action
            ),
            destination_cell_value=destination_value,
            ray_hit=(float(np.asarray(ray_hits)[step.action]) if ray_hits is not None else None),
            ray_hit_type=(float(np.asarray(ray_types)[step.action]) if ray_types is not None else None),
            chosen_score=chosen_score,
            best_safe_action=best_safe_action,
            best_safe_score=best_safe_score,
            score_margin=chosen_score - best_safe_score,
            group_attributions=attributions,
        )

    def best_safe_action(
        self,
        local_grid: np.ndarray,
        scores: np.ndarray,
        *,
        excluded_action: int | None = None,
    ) -> int:
        """Return the best safe counterfactual action."""

        safe_actions = [
            action
            for action in range(scores.shape[0])
            if action != excluded_action
            and not self.collision_visible(local_grid, action)
        ]
        if not safe_actions:
            raise ValueError("no safe counterfactual action is available")
        safe_scores = np.asarray([scores[action] for action in safe_actions], dtype=np.float32)
        return int(safe_actions[int(np.argmax(safe_scores))])

    def best_safe_action_for_observation(
        self,
        observation: Mapping[str, np.ndarray],
        scores: np.ndarray,
        *,
        excluded_action: int | None = None,
    ) -> int:
        """Return the best safe alternative for box or radial observations."""
        safe_actions = [
            action for action in range(scores.shape[0])
            if action != excluded_action
            and not self.collision_visible_observation(observation, action)
        ]
        if not safe_actions:
            raise ValueError("no safe counterfactual action is available")
        return max(safe_actions, key=lambda action: float(scores[action]))

    def destination_value(self, local_grid: np.ndarray, action: int) -> float:
        """Return the normalized categorical code at an action destination."""
        direction = self._schema.action_directions[action]
        return float(local_grid[self._schema.local_grid_index(direction)])

    def collision_visible(self, local_grid: np.ndarray, action: int) -> bool:
        """Return whether the local box marks the destination as blocked."""
        value = self.destination_value(local_grid, action)
        return bool(np.isclose(value, GEOMETRY_CELL) or np.isclose(value, TRAIL_CELL))

    def collision_visible_observation(
        self, observation: Mapping[str, np.ndarray], action: int
    ) -> bool:
        """Return immediate blocking evidence for box or radial observations."""
        if "local_grid" in observation:
            return self.collision_visible(
                np.asarray(observation["local_grid"], dtype=np.float32), action
            )
        ray_hits = np.asarray(observation["ray_hits"], dtype=np.float32)
        ray_types = np.asarray(observation["ray_hit_types"], dtype=np.float32)
        if float(ray_hits[action]) < 1.0:
            return False
        return any(
            np.isclose(float(ray_types[action]), blocking_type)
            for blocking_type in BLOCKING_RAY_TYPES
        )


class ExplanationReportWriter:
    """Write explanation reports to disk.

    Parameters
    ----------
    output_dir : Path
        Directory that receives report artifacts.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    @property
    def output_dir(self) -> Path:
        """Return artifact output directory."""

        return self._output_dir

    def write(self, report: ExplanationReport) -> Path:
        """Write report JSON and collision-step CSV artifacts."""

        self._output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._output_dir.joinpath("report.json")
        report_path.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
        self._write_collision_steps(report)
        return report_path

    def _write_collision_steps(self, report: ExplanationReport) -> None:
        """Write a compact CSV summary for selected collision steps."""

        csv_path = self._output_dir.joinpath("collision_steps.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "chosen_action",
                    "collision_visible",
                    "destination_cell_value",
                    "ray_hit",
                    "ray_hit_type",
                    "chosen_score",
                    "best_safe_action",
                    "best_safe_score",
                    "score_margin",
                ],
            )
            writer.writeheader()
            for step in report.steps:
                writer.writerow(
                    {
                        "step": step.step,
                        "chosen_action": step.chosen_action,
                        "collision_visible": step.collision_visible,
                        "destination_cell_value": step.destination_cell_value,
                        "ray_hit": step.ray_hit,
                        "ray_hit_type": step.ray_hit_type,
                        "chosen_score": step.chosen_score,
                        "best_safe_action": step.best_safe_action,
                        "best_safe_score": step.best_safe_score,
                        "score_margin": step.score_margin,
                    }
                )
