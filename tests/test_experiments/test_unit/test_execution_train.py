"""Exact-continuation and weight-only fine-tuning contracts."""

import json
from pathlib import Path

import pytest
import yaml
from theseo_anysearch.worlds.manifest import world_contract

from theseo_anysearch.experiments.execution_recipe import clone, validate
from theseo_anysearch.experiments.execution_train import train_recipe


def _fixture(tmp_path: Path, *, curriculum: bool = False) -> Path:
    checkpoint = tmp_path / "source" / "checkpoints" / "iter_000003"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text(json.dumps({
        "iteration": 3, "episodes_total": 21, "rllib_path": str(checkpoint),
        "world_contract": world_contract({"grid_size": 32})}))
    env = {"agent_count": 1, "observation": {"mode": "box", "box_radius": 1},
           "action": {"mode": "discrete_18"}, "geometry": {"grid_size": 32}}
    if curriculum:
        env["waypoint_curriculum"] = {
            "enabled": True, "initial_start": [1, 1, 1], "initial_goal": [2, 1, 1]}
    (checkpoint.parent.parent / "experiment.yaml").write_text(yaml.safe_dump({
        "experiment": {"name": "training-clone"}, "env": env,
        "training": {"algorithm": "ppo", "iterations": 10},
    }))
    return checkpoint


def test_continuation_refuses_checkpoint_without_curriculum_snapshot(tmp_path):
    recipe = clone(_fixture(tmp_path, curriculum=True), "continuation")
    report = validate(recipe)
    assert not report["execution_supported"]
    assert "curriculum/state.json" in report["execution_blocker"]
    with pytest.raises(ValueError, match="curriculum/state.json"):
        train_recipe(recipe, base=tmp_path, output_dir=tmp_path / "runs", iterations=1)


@pytest.mark.parametrize("scope,initial,final", [
    ("continuation", 3, 5), ("fine_tuning", 0, 2),
])
def test_training_scopes_have_distinct_counters_and_weight_restoration(
    tmp_path, monkeypatch, scope, initial, final,
):
    recipe = clone(_fixture(tmp_path), scope)
    built = []

    class Algorithm:
        restored = False
        weights = None
        stopped = False

        def __init__(self):
            self.learner_group = self
            self.env_runner_group = self
            self.eval_env_runner_group = None

        def restore(self, path): self.restored = True
        def get_weights(self): return {"default_policy": {"weight": 5}}
        def set_weights(self, weights): self.weights = weights
        def sync_weights(self, **kwargs):
            assert kwargs["from_worker_or_learner_group"] is self
        def stop(self): self.stopped = True

    class Trainer:
        def __init__(self, settings):
            self.settings = settings
            self._algo = None
            self._iteration = 0
            self._episodes_total = 0

        def _build_algorithm(self):
            algorithm = Algorithm()
            built.append(algorithm)
            return algorithm

        def train(self):
            assert self._iteration == initial
            assert self.settings.training.iterations == final
            self._iteration = final
            return [object()] * (final - initial)

        def checkpoint(self):
            path = self.settings.training.output_dir / "checkpoints" / f"iter_{self._iteration:06d}"
            path.mkdir(parents=True)
            return path

    monkeypatch.setattr("theseo_anysearch.rllib.algorithms.ppo.PPOTrainer.from_settings",
                        lambda settings: Trainer(settings))
    monkeypatch.setattr("theseo_anysearch.rllib.algorithms.ppo.PPOTrainer.build_algorithm_from_settings",
                        lambda settings, env_config: Algorithm())
    monkeypatch.setattr("theseo_anysearch.experiments.execution_train._policy_digest",
                        lambda algorithm: "same-policy")
    result = train_recipe(recipe, base=tmp_path, output_dir=tmp_path / "runs", iterations=2)
    record = json.loads((result / "execution.json").read_text())
    assert record["completed_iterations"] == 2
    assert record["final_iteration"] == final
    assert record["source_checkpoint_unchanged"]
    assert built[0].restored is (scope == "continuation")
    assert (built[0].weights is not None) is (scope == "fine_tuning")
    assert built[0].stopped
