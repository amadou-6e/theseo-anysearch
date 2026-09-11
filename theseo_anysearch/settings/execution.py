"""Execution-backend configuration models."""

from pydantic import BaseModel, ConfigDict, Field

class AnyscaleConfig(BaseModel):
    """Anyscale execution configuration.

    Parameters
    ----------
    cluster_env : str
        Cluster environment name.
    compute_config : str
        Compute configuration name.
    project : str
        Anyscale project identifier.
    """
    model_config = ConfigDict(extra="forbid")

    cluster_env: str = Field(description="Anyscale cluster environment name.")
    compute_config: str = Field(description="Anyscale compute configuration name.")
    project: str = Field(description="Anyscale project identifier.")
