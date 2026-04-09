"""Unit tests for cross-field experiment loader validation behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from theseo_anysearch.experiments.loader import load_experiment


def test_load_experiment_rejects_multi_agent_count_for_single_agent_ppo(tmp_path) -> None:
    """Validate load experiment rejects multi agent count for single agent ppo."""
    config_path = tmp_path / "ppo_maps.yaml"
    config_path.write_text(
        """
experiment:
  name: ppo-maps
env:
  obs_mode: radial
  agent_count: 2
training:
  algorithm: ppo
algorithm_config: {}
model_config: {}
mlflow: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="multi_agent_voxel_ppo"):
        load_experiment(config_path)


def test_load_experiment_allows_multi_agent_voxel_ppo(tmp_path) -> None:
    """Validate load experiment allows multi agent voxel ppo."""
    config_path = tmp_path / "multi_agent.yaml"
    config_path.write_text(
        """
experiment:
  name: multi-agent
env:
  obs_mode: radial
  agent_count: 2
training:
  algorithm: multi_agent_voxel_ppo
algorithm_config: {}
model_config: {}
mlflow: {}
""".strip(),
        encoding="utf-8",
    )

    experiment = load_experiment(config_path)

    assert experiment.env.agent_count == 2
    assert experiment.training.algorithm == "multi_agent_voxel_ppo"
