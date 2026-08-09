"""Typed inputs and outputs for policy explanation backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


ExplainMethod = Literal["occlusion"]
ExplainFocus = Literal["collisions", "all", "explicit"]


class ExplanationRequest(BaseModel):
    """User intent for one backend explanation run.

    Parameters
    ----------
    run_ref : str
        Registry reference such as ``"dqn-maps-zones:5932954b"``.
    checkpoint : str
        Checkpoint selector such as ``"latest"``.
    trajectory : str
        Trajectory selector such as ``"iter_000080"`` or ``"eval"``.
    method : ExplainMethod, default="occlusion"
        Attribution method requested by the caller.
    focus : ExplainFocus, default="collisions"
        Step selection strategy.
    max_steps : int, default=50
        Maximum number of selected steps to explain.
    seed : int | None, optional
        Deterministic evaluation seed.
    output_dir : Path | None, optional
        Directory for explanation artifacts.
    background : str, default="auto"
        Background observation selector.
    explicit_steps : tuple[int, ...], default=()
        Explicit step indices when ``focus`` is ``"explicit"``.
    """

    model_config = ConfigDict(frozen=True)

    run_ref: str
    checkpoint: str
    trajectory: str
    method: ExplainMethod = "occlusion"
    focus: ExplainFocus = "collisions"
    max_steps: int = 50
    seed: int | None = None
    output_dir: Path | None = None
    background: str = "auto"
    explicit_steps: tuple[int, ...] = ()
    scenario_validity: Literal[
        "environment_validated", "not_environment_validated"
    ] = "environment_validated"


class ActionScoreTable(BaseModel):
    """Action scores for a batch of observations.

    Parameters
    ----------
    values : np.ndarray
        Two-dimensional score array with shape ``(batch, action_count)``.
    score_type : str
        Human-readable score type, for example ``"q_value"``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    values: np.ndarray
    score_type: str

    def action_count(self) -> int:
        """Return the number of scored actions."""

        return int(self.values.shape[1])

    def row(self, index: int) -> np.ndarray:
        """Return one score row by batch index."""

        return self.values[index]


class ExplainedStep(BaseModel):
    """Explanation payload for one selected trace step.

    Parameters
    ----------
    step : int
        Zero-based step index in the trace.
    chosen_action : int
        Action selected by the policy or recorded in the trace.
    chosen_direction : tuple[int, int, int]
        Direction vector aligned with the 26-action voxel action space.
    collision_visible : bool
        Whether the pre-action observation showed an immediate obstruction.
    ray_hit : float
        Pre-action ``ray_hits[chosen_action]`` value.
    ray_hit_type : float
        Pre-action ``ray_hit_types[chosen_action]`` value.
    chosen_score : float
        Score for the chosen action.
    best_safe_action : int
        Highest-scoring action that is not immediately blocked.
    best_safe_score : float
        Score for ``best_safe_action``.
    score_margin : float
        ``chosen_score - best_safe_score``.
    group_attributions : dict[str, float]
        Attribution values keyed by feature group name.
    """

    model_config = ConfigDict(frozen=True)

    step: int
    chosen_action: int
    chosen_direction: tuple[int, int, int]
    collision_visible: bool
    destination_cell_value: float | None = None
    ray_hit: float | None = None
    ray_hit_type: float | None = None
    chosen_score: float
    best_safe_action: int
    best_safe_score: float
    score_margin: float
    group_attributions: dict[str, float] = Field(default_factory=dict)
    action_scores: list[float] = Field(default_factory=list)
    goal_direction: list[float] | None = None
    goal_distance: float | None = None

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable representation of this step."""

        return {
            "step": self.step,
            "chosen_action": self.chosen_action,
            "chosen_direction": list(self.chosen_direction),
            "collision_visible": self.collision_visible,
            "destination_cell_value": self.destination_cell_value,
            "ray_hit": self.ray_hit,
            "ray_hit_type": self.ray_hit_type,
            "chosen_score": self.chosen_score,
            "best_safe_action": self.best_safe_action,
            "best_safe_score": self.best_safe_score,
            "score_margin": self.score_margin,
            "group_attributions": dict(self.group_attributions),
            "action_scores": list(self.action_scores),
            "goal_direction": self.goal_direction,
            "goal_distance": self.goal_distance,
        }


class ExplanationReport(BaseModel):
    """Complete backend explanation result."""

    model_config = ConfigDict(frozen=True)

    run_ref: str
    checkpoint: str
    trajectory: str
    algorithm: str
    score_type: str
    method: str
    feature_schema_version: int
    steps: list[ExplainedStep]
    output_dir: Path | None = None
    scenario_validity: Literal["environment_validated", "not_environment_validated"] = (
        "environment_validated"
    )

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable report dictionary."""

        return {
            "run_ref": self.run_ref,
            "checkpoint": self.checkpoint,
            "trajectory": self.trajectory,
            "algorithm": self.algorithm,
            "score_type": self.score_type,
            "method": self.method,
            "feature_schema_version": self.feature_schema_version,
            "steps": [step.to_json_dict() for step in self.steps],
            "scenario_validity": self.scenario_validity,
        }
