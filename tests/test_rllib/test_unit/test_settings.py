"""Unit tests for settings loading and deep-merge behavior."""

from pathlib import Path

import pytest

from theseo_anysearch.models import AlgorithmConfig, ModelConfig
from theseo_anysearch.settings import load_settings, _deep_merge
from theseo_anysearch.rllib.algorithms.models import PPOConfig
from theseo_anysearch.rllib.models.models import VoxelEncoderConfig


class TestDeepMerge:
    """Tests DeepMerge."""
    def test_shallow(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99, "c": 3})
        assert base == {"a": 1, "b": 99, "c": 3}

    def test_nested(self):
        base = {"env": {"seed": 42, "scale": 1.0}}
        _deep_merge(base, {"env": {"seed": 7}})
        assert base["env"]["seed"] == 7
        assert base["env"]["scale"] == 1.0


class TestLoadSettings:
    """Tests LoadSettings."""
    def test_loads_minimal_yaml(self, minimal_yaml: Path):
        s = load_settings(minimal_yaml)
        assert s.env.agent_count == 1
        assert s.training.iterations == 5
        assert s.anyscale.project == "test-project"

    def test_algorithm_config_resolved_to_ppo(self, minimal_yaml: Path):
        s = load_settings(minimal_yaml)
        assert isinstance(s.algorithm_config, PPOConfig)
        assert s.algorithm_config.lr == pytest.approx(0.001)
        assert s.algorithm_config.train_batch_size == 512

    def test_model_config_resolved_to_voxel_encoder(self, minimal_yaml: Path):
        s = load_settings(minimal_yaml)
        assert isinstance(s.model_cfg, VoxelEncoderConfig)
        assert s.model_cfg.hidden_sizes == [128]
        assert s.model_cfg.activation == "relu"

    def test_unknown_algorithm_raises(self, tmp_path: Path, minimal_yaml: Path):
        import yaml as _yaml
        raw = _yaml.safe_load(minimal_yaml.read_text())
        raw["training"]["algorithm"] = "unknown_algo"
        p = tmp_path / "s2.yaml"
        p.write_text(_yaml.dump(raw))
        with pytest.raises(ValueError, match="unknown_algo"):
            load_settings(p)

    def test_override_applied(self, minimal_yaml: Path):
        s = load_settings(minimal_yaml, overrides={"env": {"seed": 999}})
        assert s.env.seed == 999

    def test_nested_override(self, minimal_yaml: Path):
        s = load_settings(minimal_yaml, overrides={"training": {"iterations": 50}})
        assert s.training.iterations == 50
        assert s.training.algorithm == "ppo"

    def test_missing_model_uses_default(self, tmp_path: Path, minimal_yaml: Path):
        import yaml as _yaml

        raw = _yaml.safe_load(minimal_yaml.read_text())
        raw["training"].pop("model")
        path = tmp_path.joinpath("settings_missing_model.yaml")
        path.write_text(_yaml.dump(raw))

        settings = load_settings(path)
        assert settings.training.model == "voxel_encoder"

    def test_missing_anyscale_allowed_for_local_runner(self, tmp_path: Path, minimal_yaml: Path):
        import yaml as _yaml

        raw = _yaml.safe_load(minimal_yaml.read_text())
        raw.pop("anyscale")
        path = tmp_path.joinpath("settings_local_no_anyscale.yaml")
        path.write_text(_yaml.dump(raw))

        settings = load_settings(path)
        assert settings.anyscale.project == ""
def test_training_accepts_remote_learner_resources(minimal_yaml: Path) -> None:
    from theseo_anysearch.settings import load_settings

    content = minimal_yaml.read_text(encoding="utf-8")
    content = content.replace(
        "algorithm: ppo",
        "algorithm: ppo\n  num_learners: 1\n  num_cpus_per_learner: 1\n  num_gpus_per_learner: 0.3333333333333333",
    )
    minimal_yaml.write_text(content, encoding="utf-8")

    settings = load_settings(minimal_yaml)

    assert settings.training.num_learners == 1
    assert settings.training.num_cpus_per_learner == 1
    assert settings.training.num_gpus_per_learner == pytest.approx(1 / 3)
