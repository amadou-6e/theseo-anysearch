"""Validated configuration and artifacts for heuristic imitation learning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class ProviderSelector(BaseModel):
    """Select a named Python or Rust imitation provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


def _expand_provider_shorthand(value: Any) -> Any:
    """Accept ``provider: name`` as shorthand for the selector block."""
    if isinstance(value, str):
        return {"name": value}
    return value


class GenerationConfig(BaseModel):
    """Controls deterministic episode-generation provider rollout."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderSelector = Field(default_factory=lambda: ProviderSelector(name="astar"))
    episodes: int = Field(default=500, ge=2)
    max_attempts: int = Field(default=1000, ge=1)
    require_success: bool = True

    @field_validator("provider", mode="before")
    @classmethod
    def expand_provider_shorthand(cls, value: Any) -> Any:
        return _expand_provider_shorthand(value)

    @model_validator(mode="after")
    def validate_attempt_budget(self) -> "GenerationConfig":
        if self.max_attempts < self.episodes:
            raise ValueError(
                "imitation.generation.max_attempts must be at least episodes"
            )
        return self


class SamplingConfig(BaseModel):
    """Selects the batch-sampling provider used during pretraining."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderSelector = Field(
        default_factory=lambda: ProviderSelector(name="uniform_transition")
    )

    @field_validator("provider", mode="before")
    @classmethod
    def expand_provider_shorthand(cls, value: Any) -> Any:
        return _expand_provider_shorthand(value)


class DemonstrationCollectionConfig(BaseModel):
    """Controls deterministic dataset collection and reuse."""

    model_config = ConfigDict(extra="forbid")

    seed_start: int = Field(default=1000, ge=0)
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    reuse_dataset: bool = True
    dataset_dir: str | None = None
    curriculum_stages: Literal["initial", "all"] = "initial"


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
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
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

    schema_version: Literal[1, 2, 3] = 3
    fingerprint: str
    generation_provider_name: str
    generation_provider_parameters: dict[str, JsonValue]
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
    stage_episode_counts: list[int] | None = None


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
