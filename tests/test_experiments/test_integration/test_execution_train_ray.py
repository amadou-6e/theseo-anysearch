"""Real RLlib smoke for both checkpoint training scopes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from theseo_anysearch.experiments.execution_recipe import (
    ExecutionRecipe, GeometryDecision, clone, make_portable,
)
from theseo_anysearch.experiments.execution_evaluate import evaluate
from theseo_anysearch.experiments.execution_train import train_recipe
from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.experiments.output import OutputStore
from theseo_anysearch.experiments.loader import _resolve_typed_configs
from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer
from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent

pytestmark = pytest.mark.ray


@pytest.mark.timeout(240)
def test_real_continuation_and_weight_only_fine_tuning(tmp_path: Path) -> None:
    source = tmp_path / "source"
    raw = {
        "experiment": {"name": "checkpoint-clone"},
        "env": {"agent_count": 1, "max_steps": 8, "seed": 0,
                "geometry": {"grid_size": 32, "boxes": []},
                "observation": {"mode": "box", "box_radius": 2},
                "waypoint_curriculum": {
                    "enabled": True, "initial_start": [1, 1, 1], "initial_goal": [3, 1, 1],
                }},
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

    original_eval = evaluate(clone(checkpoint, "evaluation"), base=tmp_path,
                             output_dir=tmp_path / "original-eval", episodes=2, seed=13)
    portable = make_portable(clone(checkpoint, "evaluation"), tmp_path / "bundle")
    portable.save(tmp_path / "bundle" / "recipe.yaml")
    (tmp_path / "bundle").rename(tmp_path / "moved-bundle")
    moved = ExecutionRecipe.load(tmp_path / "moved-bundle" / "recipe.yaml")
    moved_eval = evaluate(moved, base=tmp_path / "moved-bundle",
                          output_dir=tmp_path / "moved-eval", episodes=2, seed=13)
    assert json.loads((original_eval / "metrics.json").read_text()) == json.loads(
        (moved_eval / "metrics.json").read_text())
    for index in range(2):
        path = f"trajectories/episode_{index:06d}.json.zst"
        original_episode = OutputStore(original_eval).read_json(path)["episode"]
        moved_episode = OutputStore(moved_eval).read_json(path)["episode"]
        assert original_episode == moved_episode

    compiled = compile_world([BoxSource((8, 8, 8), (8, 8, 8))],
                             WorldExtent(x=32, y=32, z=32), tmp_path / "replacement")
    moved.overrides.task = GeometryDecision.preserve()
    moved.overrides.routes = GeometryDecision.preserve()
    replacement_eval = evaluate(moved, base=tmp_path / "moved-bundle",
                                output_dir=tmp_path / "replacement-eval",
                                episodes=1, seed=13,
                                world=compiled.root / "manifest.json")
    difference = json.loads((replacement_eval / "difference_manifest.json").read_text())
    assert difference["changes"][0]["to"] == compiled.manifest.identity_sha256
    assert json.loads((replacement_eval / "metrics.json").read_text())["evaluated_episodes"] == 1

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
