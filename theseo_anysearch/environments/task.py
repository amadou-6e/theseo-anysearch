"""Versioned task, goal, reward, and termination contracts."""

from __future__ import annotations

from math import dist
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Coord = tuple[int, int, int]


class PointGoal(BaseModel):
    """A single voxel that must be occupied by the cursor."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["point"] = "point"
    position: Coord | None = None
    tolerance: float = Field(default=0.0, ge=0.0)


class TargetVoxelSetGoal(BaseModel):
    """A goal satisfied by entering any voxel in a target set."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["target_voxel_set"] = "target_voxel_set"
    voxels: list[Coord] = Field(min_length=1)


Goal = Annotated[PointGoal | TargetVoxelSetGoal, Field(discriminator="type")]


class TerminationPolicy(BaseModel):
    """Conditions that end an episode as a task termination."""

    model_config = ConfigDict(extra="forbid")
    terminate_on_success: bool = True


class TaskConfig(BaseModel):
    """YAML-facing versioned task definition."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    max_consecutive_collisions: int | None = Field(default=None, ge=1)
    goal: Goal = Field(default_factory=PointGoal)
    termination: TerminationPolicy = Field(default_factory=TerminationPolicy)
    construction_target_voxels: list[Coord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_construction_target(self) -> "TaskConfig":
        if (
            self.construction_target_voxels
            and isinstance(self.goal, TargetVoxelSetGoal)
            and set(self.construction_target_voxels) != set(self.goal.voxels)
        ):
            raise ValueError(
                "construction_target_voxels and target_voxel_set goal must describe the same voxels"
            )
        return self


def goal_voxels(goal: Goal, fallback: Coord | None) -> tuple[Coord, ...]:
    """Resolve a configured goal to concrete target voxels."""

    if isinstance(goal, TargetVoxelSetGoal):
        return tuple(goal.voxels)
    if goal.position is not None:
        return (goal.position,)
    return () if fallback is None else (fallback,)


def goal_distance(goal: Goal, cursor: Coord, fallback: Coord | None) -> float:
    """Return Euclidean distance to the nearest accepted voxel."""

    targets = goal_voxels(goal, fallback)
    return min((dist(cursor, target) for target in targets), default=0.0)


def is_success(goal: Goal, cursor: Coord, fallback: Coord | None) -> bool:
    """Evaluate success independently of reward magnitude."""

    if isinstance(goal, PointGoal):
        targets = goal_voxels(goal, fallback)
        return bool(targets) and dist(cursor, targets[0]) <= goal.tolerance
    return cursor in set(goal.voxels)
