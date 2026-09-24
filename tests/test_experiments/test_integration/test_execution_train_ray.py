"""Real RLlib smoke for both checkpoint training scopes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from theseo_anysearch.experiments.execution_recipe import clone
from theseo_anysearch.experiments.execution_train import train_recipe
from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.experiments.loader import _resolve_typed_configs
from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer

pytestmark = pytest.mark.ray


@pytest.mark.timeout(240)
def test_real_continuation_and_weight_only_fine_tuning(tmp_path: Path) -> None:
    source = tmp_path / "source"
    raw = {
        "experiment": {"name": "checkpoint-clone"},
        "env": {"agent_count": 1, "max_steps": 8, "seed": 0,
                "geometry": {"grid_size": 32, "boxes": []},
                "observation": {"mode": "box", "box_radius": 2}},
        "training": {"algorithm": "ppo", "iterations": 1,
                     "checkpoint_interval": 1, "output_dir": str(source),
                     "video_every": 0, "num_env_runners": 0},
        "evaluation": {"enabled": False},
        "algorithm_config": {"train_batch_size": 32, "minibatch_size": 16,
                             "num_sgd_iter": 1},
        "model_config": {"hidden_sizes": [32], "activation": "relu"},
    }
    config = ExperimentConfig.model_validate(_resolve_typed_configs(raw))
    source.mkdir()
    (source / "experiment.yaml").write_text(yaml.safe_dump(raw))
    trainer = PPOTrainer(config.to_settings())
    trainer.train()
    trainer._algo.stop()
    checkpoint = source / "checkpoints" / "iter_000001"
    assert checkpoint.is_dir()

    continued = train_recipe(clone(checkpoint, "continuation"), base=tmp_path,
                             output_dir=tmp_path / "continued", iterations=1)
    tuned = train_recipe(clone(checkpoint, "fine_tuning"), base=tmp_path,
                         output_dir=tmp_path / "tuned", iterations=1)
    continuation = json.loads((continued / "execution.json").read_text())
    fine_tuning = json.loads((tuned / "execution.json").read_text())
    assert continuation["source_iteration"] == 1
    assert continuation["final_iteration"] == 2
    assert fine_tuning["source_iteration"] == 1
    assert fine_tuning["final_iteration"] == 1
    assert "optimizer" in fine_tuning["resets"]
    assert "optimizer" not in continuation["resets"]
    assert continuation["source_policy_sha256"] == continuation["initial_policy_sha256"]
    assert fine_tuning["source_policy_sha256"] == fine_tuning["initial_policy_sha256"]
