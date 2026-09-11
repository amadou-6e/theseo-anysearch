"""Unit tests for sweep expansion behavior."""

from __future__ import annotations

from pathlib import Path

from theseo_anysearch.experiments.loader import expand_sweep, load_experiment


class TestSweepConfigExtra:
    """Verify base config inheritance across expanded sweeps."""

    def test_base_gamma_preserved_in_each_entry(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        for entry in entries:
            assert entry.algorithm_config.gamma == 0.99

    def test_base_env_preserved_in_each_entry(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        for entry in entries:
            assert entry.env.agent_count == 1

    def test_empty_experiments_list(self, tmp_path: Path):
        yaml_text = """\
sweep:
  base:
    experiment:
      name: empty-sweep
      output_dir: /tmp/out
      seed: 1
    env:
      stl_path: /tmp/test.stl
      scale: 1.0
      agent_count: 1
      max_steps: 50
    training:
      algorithm: ppo
      model: voxel_encoder
      runner: local
      iterations: 2
      checkpoint_interval: 1
      output_dir: /tmp/out
      video_every: 10
    algorithm_config:
      lr: 0.001
      gamma: 0.99
      train_batch_size: 64
    model_config:
      hidden_sizes: [64]
      activation: relu
  experiments: []
"""
        path = tmp_path.joinpath("empty_sweep.yaml")
        path.write_text(yaml_text, encoding="utf-8")
        sweep = load_experiment(path)
        assert expand_sweep(sweep) == []

    def test_each_entry_experiment_name_set(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        for entry in entries:
            assert entry.experiment.name != ""
