"""Shared fixtures for experiment unit tests.

Examples
--------
Load a single experiment config for runner or model tests.

>>> config = load_experiment(single_yaml)
>>> isinstance(config, ExperimentConfig)
True
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from theseo_anysearch.experiments.loader import load_experiment
from theseo_anysearch.experiments.models import ExperimentConfig


SINGLE_YAML = textwrap.dedent("""\
    experiment:
      name: test-run
      output_dir: {output_dir}
      seed: 7

    env:
      stl_path: /tmp/test.stl
      scale: 1.0
      agent_count: 1
      max_steps: 50
      seed: 0

    training:
      algorithm: ppo
      model: voxel_encoder
      runner: local
      iterations: 3
      checkpoint_interval: 1
      output_dir: {output_dir}
      video_every: 10

    algorithm_config:
      lr: 0.001
      gamma: 0.99
      train_batch_size: 64
      clip_param: 0.2
      num_sgd_iter: 2
      lambda_: 0.95
      kl_coeff: 0.2

    model_config:
      hidden_sizes: [64]
      activation: relu
""")

SWEEP_YAML = textwrap.dedent("""\
    sweep:
      base:
        experiment:
          name: sweep-base
          output_dir: {output_dir}
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
          output_dir: {output_dir}
          video_every: 10
        algorithm_config:
          lr: 0.001
          gamma: 0.99
          train_batch_size: 64
        model_config:
          hidden_sizes: [64]
          activation: relu

      experiments:
        - name: sweep-lr-low
          algorithm_config:
            lr: 1.0e-4
        - name: sweep-lr-high
          algorithm_config:
            lr: 1.0e-2
""")


@pytest.fixture()
def single_yaml(tmp_path: Path) -> Path:
    """Return a single-run experiment YAML path."""

    path = tmp_path.joinpath("experiment.yaml")
    path.write_text(SINGLE_YAML.format(output_dir=str(tmp_path)), encoding="utf-8")
    return path


@pytest.fixture()
def sweep_yaml(tmp_path: Path) -> Path:
    """Return a sweep experiment YAML path."""

    path = tmp_path.joinpath("sweep.yaml")
    path.write_text(SWEEP_YAML.format(output_dir=str(tmp_path)), encoding="utf-8")
    return path


@pytest.fixture()
def experiment_config(single_yaml: Path) -> ExperimentConfig:
    """Return a validated experiment config for unit tests."""

    result = load_experiment(single_yaml)
    assert isinstance(result, ExperimentConfig)
    return result
