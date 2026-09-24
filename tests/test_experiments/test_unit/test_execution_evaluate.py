"""Inference execution invariants for checkpoint recipes."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from theseo_anysearch.experiments.execution_evaluate import _policy_digest, _rebind_artifacts, evaluate
from theseo_anysearch.experiments.execution_recipe import Artifact, ExecutionRecipe, clone


def _recipe() -> ExecutionRecipe:
    artifact = Artifact(role="rllib_checkpoint", path="artifacts/checkpoint", sha256="0" * 64)
    experiment = Artifact(role="resolved_experiment", path="artifacts/experiment.yaml", sha256="1" * 64)
    return ExecutionRecipe(scope="evaluation", source_run="old", checkpoint=artifact,
                           experiment=experiment, checkpoint_state={}, policy_contract={})


def test_policy_digest_detects_mutation():
    class Tensor:
        def __init__(self):
            self.value = np.array([1.0, 2.0], dtype=np.float32)

        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def numpy(self): return self.value

    tensor = Tensor()
    algorithm = SimpleNamespace(env_runner=SimpleNamespace(
        module=SimpleNamespace(state_dict=lambda: {"weight": tensor})))
    first = _policy_digest(algorithm)
    assert first == _policy_digest(algorithm)
    tensor.value[0] = 3
    assert first != _policy_digest(algorithm)


def test_portable_catalog_and_extension_are_rebound(tmp_path: Path):
    recipe = _recipe()
    catalog = tmp_path / "artifacts" / "catalog" / "catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}")
    recipe.assets = [Artifact(role="geometry_catalog", path="artifacts/catalog", sha256="2" * 64)]
    recipe.extension = [Artifact(role="extension_manifest", path="artifacts/native/extension.json",
                                 sha256="3" * 64)]
    env = {"compiled_world_catalog_path": "runtime/old/catalog.json",
           "compiled_world_path": "C:/old/world"}
    _rebind_artifacts(recipe, env, tmp_path, False)
    assert env["compiled_world_catalog_path"] == str(catalog.resolve())
    assert env["compiled_world_path"] is None
    assert env["native_extension_manifest"] == str((tmp_path / "artifacts/native/extension.json").resolve())


def test_missing_archived_geometry_fails_closed(tmp_path: Path):
    env = {"compiled_world_catalog_path": "runtime/missing/catalog.json"}
    with pytest.raises(ValueError, match="not bundled"):
        _rebind_artifacts(_recipe(), env, tmp_path, False)


def test_evaluate_restores_without_training_and_writes_fresh_result(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "source" / "checkpoints" / "iter_000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text("{}")
    (checkpoint.parent.parent / "experiment.yaml").write_text(yaml.safe_dump({
        "experiment": {"name": "clone-test"},
        "env": {"agent_count": 1, "observation": {"mode": "box", "box_radius": 1},
                "action": {"mode": "discrete_18"}, "geometry": {"grid_size": 32}},
        "training": {"algorithm": "ppo"},
    }))
    recipe = clone(checkpoint, "evaluation")

    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def numpy(self): return np.array([5.0], dtype=np.float32)

    class Algorithm:
        env_runner = SimpleNamespace(module=SimpleNamespace(state_dict=lambda: {"weight": Tensor()}))
        restored = None
        stopped = False

        def restore(self, path): self.restored = path
        def stop(self): self.stopped = True
        def train(self): raise AssertionError("learner must not run")

    algorithm = Algorithm()
    monkeypatch.setattr("theseo_anysearch.rllib.algorithms.ppo.PPOTrainer.build_algorithm_from_settings",
                        lambda settings, env_config: algorithm)
    monkeypatch.setattr("theseo_anysearch.experiments.execution_evaluate.collect_eval_episodes",
                        lambda algo, env, count, seed: [SimpleNamespace(total_reward=1.0)])
    monkeypatch.setattr("theseo_anysearch.experiments.execution_evaluate.EvaluationMetrics.from_voxel_episodes",
                        lambda episodes, env, min_success_rate: SimpleNamespace(model_dump=lambda **kw: {"successes": 1}))
    monkeypatch.setattr("theseo_anysearch.experiments.execution_evaluate.TrajectoryWriter.write_episode",
                        lambda self, path, episode, **kw: self._store.write_bytes(path, b"episode"))

    result = evaluate(recipe, base=tmp_path, output_dir=tmp_path / "results", episodes=1, seed=9)
    assert algorithm.restored == str(checkpoint.resolve())
    assert algorithm.stopped
    assert (result / "trajectories/episode_000000.json.zst").is_file()
    assert (result / "runtime_env_config.json").is_file()
    assert (result / "metrics.json").is_file()
    assert __import__("json").loads((result / "execution.json").read_text())["policy_unchanged"]
