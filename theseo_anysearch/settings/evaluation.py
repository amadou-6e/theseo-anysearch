"""Evaluation configuration models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from theseo_anysearch.settings.environment.curriculum import WaypointAdvanceConfig
from theseo_anysearch.settings.environment.scenarios import ScenarioConfig


class WaypointCurriculumEvaluationConfig(BaseModel):
    """Retention evaluation over visited waypoint stages."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    frequency: int = Field(default=1, ge=1)
    episodes: int = Field(default=1, ge=1)
    stages: Literal["all"] = "all"
    min_success_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    min_per_stage_success_rate: float = Field(default=0.67, ge=0.0, le=1.0)
    advance: WaypointAdvanceConfig | None = None


class EvaluationConfig(BaseModel):
    """Deterministic policy evaluation and RLlib evaluation-worker settings."""

    model_config = ConfigDict(extra="forbid")

    frequency: int = Field(default=1, ge=1, description="Run evaluation every N training iterations.")
    episodes: int = Field(default=1, ge=1, description="Number of deterministic evaluation episodes.")
    seed: int = Field(42, description="Base seed for deterministic evaluation episodes.")
    min_success_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Success-rate threshold used by evaluation gates.")
    num_env_runners: int = Field(default=0, ge=0, description="Dedicated RLlib evaluation workers.")
    num_envs_per_env_runner: int = Field(default=1, ge=1, description="Vectorized environments hosted by each evaluation worker.")
    waypoint_curriculum: WaypointCurriculumEvaluationConfig = Field(
        default_factory=WaypointCurriculumEvaluationConfig,
        description="Retention evaluation and optional evaluation-gated advancement.",
    )
    scenarios: ScenarioConfig = Field(
        default_factory=ScenarioConfig,
        description="Optional deterministic scenario provider used only for evaluation.",
    )
    parallel_to_training: bool = Field(
        default=False,
        description="Run deterministic evaluation concurrently with the training update.",
    )

    @model_validator(mode="after")
    def validate_parallel_evaluation_workers(self) -> "EvaluationConfig":
        """Require dedicated actors when evaluation overlaps training."""
        if self.parallel_to_training and self.num_env_runners < 1:
            raise ValueError(
                "evaluation.parallel_to_training requires "
                "evaluation.num_env_runners >= 1"
            )
        return self
