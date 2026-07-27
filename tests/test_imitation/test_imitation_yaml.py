"""Validation tests for the documented imitation experiment YAML."""

from pathlib import Path

from theseo_anysearch.experiments.loader import load_experiment


def test_tiny_overfit_imitation_yaml_loads_and_reaches_trainer_settings():
    config = load_experiment(
        Path(
            "usage",
            "experiments",
            "train",
            "ppo_tiny_overfit_imitation.yaml",
        )
    )

    assert config.imitation.enabled is True
    assert config.imitation.teacher.type == "astar"
    assert config.imitation.collection.episodes == 64
    assert config.to_settings().imitation == config.imitation
