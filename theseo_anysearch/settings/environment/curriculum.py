"""Waypoint curriculum configuration models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WaypointAdvanceConfig(BaseModel):
    """Condition used to advance a waypoint curriculum stage."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["fixed", "interval", "success"] = "fixed"
    interval_iterations: int = Field(default=10, ge=1)
    require_success: bool = True
    successes_required: int = Field(default=1, ge=1)


class WaypointTrainingSamplingConfig(BaseModel):
    """Select how visited curriculum stages are sampled for training."""

    model_config = ConfigDict(extra="forbid")
    strategy: str = "legacy"
    current_stage_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    retained_stage_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    latest_multiplier: float = Field(default=10.0, gt=0.0)
    recency_decay: float = Field(default=0.7, gt=0.0, le=1.0)
    minimum_weight: float = Field(default=0.1, gt=0.0)
    power: float = Field(default=1.0, gt=0.0)
    unevaluated_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "WaypointTrainingSamplingConfig":
        if not self.strategy.strip():
            raise ValueError("waypoint training sampling strategy cannot be empty")
        if self.strategy == "legacy":
            total = self.current_stage_probability + self.retained_stage_probability
            if abs(total - 1.0) > 1e-9:
                raise ValueError("waypoint training sampling probabilities must sum to 1.0")
        return self


class WaypointDifficultyConfig(BaseModel):
    """Control how successive waypoint difficulty is sampled."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["random", "monotonic_distance", "goal_distance", "segment_distance"] = "random"
    initial_distance: int | None = Field(default=None, ge=1)
    distance_increment: float = Field(default=1.0, gt=0.0)
    maximum_distance: float | None = Field(default=None, gt=0.0)
    sampling_attempts: int = Field(default=512, ge=1)


class WaypointRouteLengthConfig(BaseModel):
    """Exact shortest-path length requested for a multi-waypoint route."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["fixed", "fraction"]
    distance: int | None = Field(default=None, ge=1)
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_mode_value(self) -> "WaypointRouteLengthConfig":
        if self.mode == "fixed":
            if self.distance is None or self.fraction is not None:
                raise ValueError("fixed route_length requires distance only")
        elif self.fraction is None or self.distance is not None:
            raise ValueError("fraction route_length requires fraction only")
        return self

    def resolve(self, max_steps: int) -> int:
        if self.mode == "fixed":
            assert self.distance is not None
            return self.distance
        assert self.fraction is not None
        return max(1, int(max_steps * self.fraction))


class WaypointCurriculumConfig(BaseModel):
    """Curriculum of reproducible start/goal stages or waypoint routes."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    completion_mode: Literal["terminate_episode", "continue_route"] = "terminate_episode"
    initial_start: tuple[int, int, int] | None = None
    initial_goal: tuple[int, int, int] | None = None
    route_length: WaypointRouteLengthConfig | None = None
    seed: int = 42
    difficulty: WaypointDifficultyConfig = Field(default_factory=WaypointDifficultyConfig)
    training_sampling: WaypointTrainingSamplingConfig = Field(default_factory=WaypointTrainingSamplingConfig)
    advance: WaypointAdvanceConfig = Field(default_factory=WaypointAdvanceConfig)

    @model_validator(mode="after")
    def validate_initial_pair(self) -> "WaypointCurriculumConfig":
        if self.enabled and self.completion_mode == "terminate_episode" and (
            self.initial_start is None or self.initial_goal is None
        ):
            raise ValueError("enabled waypoint_curriculum requires initial_start and initial_goal")
        if self.completion_mode == "continue_route":
            if self.enabled and self.initial_start is None:
                raise ValueError("enabled waypoint_curriculum requires initial_start")
            if self.route_length is None:
                raise ValueError("continue_route requires route_length")
            if self.difficulty.mode != "segment_distance":
                raise ValueError("continue_route requires difficulty.mode: segment_distance")
            if self.difficulty.initial_distance is None:
                raise ValueError("segment_distance requires initial_distance")
        elif self.route_length is not None:
            raise ValueError("route_length is only valid with continue_route")
        if (
            self.initial_start is not None
            and self.initial_goal is not None
            and self.difficulty.mode == "monotonic_distance"
            and self.difficulty.maximum_distance is not None
        ):
            initial_distance = sum(
                (goal - start) ** 2
                for start, goal in zip(self.initial_start, self.initial_goal)
            ) ** 0.5
            if self.difficulty.maximum_distance < initial_distance:
                raise ValueError(
                    "difficulty.maximum_distance cannot be below the initial waypoint distance"
                )
        return self
