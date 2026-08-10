"""Validated configuration and artifacts for heuristic imitation learning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImitationTeacherConfig(BaseModel):
    """Heuristic that labels demonstration observations."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["astar", "dijkstra", "weighted_astar", "replanning_astar"] = "astar"
    weight: float | None = None

    @model_validator(mode="after")
    def validate_weight(self) -> "ImitationTeacherConfig":
        if self.type == "weighted_astar":
            if self.weight is None:
                self.weight = 1.5
            elif self.weight <= 0.0:
                raise ValueError("imitation.teacher.weight must be greater than zero")
        elif self.weight is not None:
            raise ValueError(
                "imitation.teacher.weight is only valid for weighted_astar"
            )
        return self


class DemonstrationCollectionConfig(BaseModel):
    """Controls deterministic teacher rollout collection."""

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(default=500, ge=2)
    seed_start: int = Field(default=1000, ge=0)
    max_attempts: int = Field(default=1000, ge=1)
    require_success: bool = True
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    reuse_dataset: bool = True
    dataset_dir: str | None = None

    @model_validator(mode="after")
    def validate_attempt_budget(self) -> "DemonstrationCollectionConfig":
        if self.max_attempts < self.episodes:
            raise ValueError(
                "imitation.collection.max_attempts must be at least episodes"
            )
        return self


class ImitationPretrainingConfig(BaseModel):
    """Behavior-cloning optimizer settings."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    early_stopping_patience: int = Field(default=3, ge=1)


class ImitationHandoffConfig(BaseModel):
    """Selects which behavior-cloned parameters PPO receives."""

    model_config = ConfigDict(extra="forbid")

    initialize_encoder: bool = True
    initialize_policy: bool = True
    initialize_value_head: bool = False

    @model_validator(mode="after")
    def validate_any_handoff(self) -> "ImitationHandoffConfig":
        if not (
            self.initialize_encoder
            or self.initialize_policy
            or self.initialize_value_head
        ):
            raise ValueError("imitation.handoff must initialize at least one model part")
        return self


class ImitationCacheConfig(BaseModel):
    """Content-addressed behavior-cloning cache settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    directory: str | None = None
    refresh: bool = False
    lock_timeout_seconds: float = Field(default=1800.0, gt=0.0)


class ImitationConfig(BaseModel):
    """Top-level heuristic imitation stage configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    strategy: Literal["pretrain_then_rl"] = "pretrain_then_rl"
    teacher: ImitationTeacherConfig = Field(default_factory=ImitationTeacherConfig)
    collection: DemonstrationCollectionConfig = Field(
        default_factory=DemonstrationCollectionConfig
    )
    pretraining: ImitationPretrainingConfig = Field(
        default_factory=ImitationPretrainingConfig
    )
    handoff: ImitationHandoffConfig = Field(default_factory=ImitationHandoffConfig)
    cache: ImitationCacheConfig = Field(default_factory=ImitationCacheConfig)


class DemonstrationManifest(BaseModel):
    """Versioned identity and summary for a collected dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    fingerprint: str
    teacher_type: str
    teacher_weight: float | None
    requested_episodes: int
    successful_episodes: int
    accepted_episodes: int
    attempted_episodes: int
    training_episodes: int
    validation_episodes: int
    training_samples: int
    validation_samples: int
    observation_size: int
    action_count: int
    action_nvec: list[int] | None = None
    seeds: list[int]


class ImitationResult(BaseModel):
    """Serializable behavior-cloning result reported by the trainer."""

    model_config = ConfigDict(extra="forbid")

    epochs_completed: int
    best_validation_loss: float
    validation_accuracy: float
    training_samples: int
    validation_samples: int
    checkpoint_path: str
    pre_rl_success_rate: float | None = None
    cache_hit: bool = False
    cache_key: str | None = None
    policy_id: str = "default_policy"
