"""Base algorithm and model configuration contracts."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

class AlgorithmConfig(BaseModel):
    """Base algorithm hyperparameters shared by multiple trainers.

    Parameters
    ----------
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    train_batch_size : int
        Training batch size passed into RLlib.
    """
    model_config = ConfigDict(extra="forbid")

    lr: float = Field(3e-4, gt=0.0, description="Optimizer learning rate.")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor for future rewards.")
    train_batch_size: int = Field(4096, ge=1, description="Samples consumed by each training update.")


class ModelConfig(BaseModel):
    """Base model configuration shared by trainer integrations.

    Parameters
    ----------
    hidden_sizes : list[int]
        Hidden layer sizes for the default fully connected policy.
    activation : {"relu", "tanh", "elu"}
        Hidden activation used by the default policy.
    custom_model : str | None
        Registered RLlib custom model name.
    custom_model_config : dict[str, Any] | None
        Extra configuration passed to the custom model.
    """
    model_config = ConfigDict(extra="forbid")

    hidden_sizes: list[Annotated[int, Field(gt=0)]] = Field(default_factory=lambda: [256, 256], description="Hidden layer widths of the default fully connected policy.")
    activation: Literal["relu", "tanh", "elu"] = Field("relu", description="Hidden-layer activation function.")
    custom_model: str | None = Field(None, description="Registered RLlib model implementation name.")
    custom_model_config: dict[str, Any] | None = Field(None, description="Parameters forwarded to the registered model implementation.")


class AlgorithmEnvCompatibilityMixin:
    """Validate algorithm and environment combinations after model creation."""

    @model_validator(mode="after")
    def _validate_algorithm_env_compatibility(self):
        """Reject unsupported algorithm and agent-count combinations."""
        threshold = self.training.early_stop.min_goal_finishes
        if threshold is not None and threshold > self.evaluation.episodes:
            raise ValueError(
                "training.early_stop.min_goal_finishes cannot exceed evaluation.episodes"
            )
        single_agent_algorithms = {"ppo", "appo", "dqn", "sac", "rainbow"}
        algorithm = self.training.algorithm.lower()
        if (
            self.training.early_stop.enabled
            and self.training.early_stop.mode in {"heuristic_accuracy", "heuristic_distance"}
            and self.env.agent_count != 1
        ):
            raise ValueError(
                "heuristic comparison early stopping requires env.agent_count: 1"
            )
        if algorithm in single_agent_algorithms and self.env.agent_count != 1:
            raise ValueError(
                f"training.algorithm='{self.training.algorithm}' only supports single-agent "
                f"VoxelEnv, but env.agent_count={self.env.agent_count}. "
                "Use env.agent_count=1 or switch to training.algorithm='multi_agent_voxel_ppo'."
            )
        return self
