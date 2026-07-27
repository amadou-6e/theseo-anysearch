"""Root validated runtime settings."""

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.imitation.models import ImitationConfig
from theseo_anysearch.settings.algorithm import AlgorithmConfig, AlgorithmEnvCompatibilityMixin, ModelConfig
from theseo_anysearch.settings.environment import EnvConfig
from theseo_anysearch.settings.evaluation import EvaluationConfig
from theseo_anysearch.settings.execution import AnyscaleConfig
from theseo_anysearch.settings.training import TrainingConfig

class Settings(AlgorithmEnvCompatibilityMixin, BaseModel):
    """Validated runtime settings consumed by trainers and runners.

    Parameters
    ----------
    env : EnvConfig
        Environment configuration.
    training : TrainingConfig
        Training execution configuration.
    anyscale : AnyscaleConfig
        Anyscale execution configuration.
    algorithm_config : AlgorithmConfig
        Algorithm-specific hyperparameter block.
    model_cfg : ModelConfig
        Model configuration block loaded from ``model_config`` in YAML.
    """
    # Pydantic v2 reserves 'model_config' as a class variable.
    # The YAML key is 'model_config'; stored as 'model_cfg' with an alias.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    env: EnvConfig = Field(description="Environment configuration.")
    training: TrainingConfig = Field(description="Training execution configuration.")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Deterministic evaluation configuration.")
    anyscale: AnyscaleConfig = Field(description="Anyscale execution configuration.")
    algorithm_config: AlgorithmConfig = Field(description="Algorithm-specific hyperparameters.")
    model_cfg: ModelConfig = Field(alias="model_config", description="Policy model configuration loaded from model_config.")
    imitation: ImitationConfig = Field(default_factory=ImitationConfig)
