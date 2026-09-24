"""Inference execution invariants for checkpoint recipes."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from theseo_anysearch.experiments.execution_evaluate import (
    _policy_digest, _rebind_artifacts, _safe_output_root,
    _verify_policy_spaces, _verify_replacement_world, evaluate,
)
from theseo_anysearch.experiments.execution_recipe import Artifact, ExecutionRecipe, clone
from theseo_anysearch.worlds.compiler import BoxSource, WorldPackCorruptError, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent


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


def test_output_cannot_overlap_bundle_or_checkpoint(tmp_path: Path):
    recipe = _recipe()
    with pytest.raises(ValueError, match="portable recipe bundle"):
        _safe_output_root(recipe, tmp_path / "bundle", tmp_path / "bundle" / "runs")
    recipe.checkpoint.path = str(tmp_path / "checkpoint")
    (tmp_path / "checkpoint").mkdir()
    recipe.experiment.path = str(tmp_path / "experiment.yaml")
    with pytest.raises(ValueError, match="rllib_checkpoint"):
        _safe_output_root(recipe, tmp_path, tmp_path / "checkpoint" / "runs")


def test_replacement_world_pack_is_verified_before_execution(tmp_path: Path):
    compiled = compile_world([BoxSource((2, 2, 2), (2, 2, 2))],
                             WorldExtent(x=8, y=8, z=8), tmp_path / "compiled")
    manifest = compiled.root / "manifest.json"
    _verify_replacement_world(_recipe(), manifest, tmp_path)
    compiled.pack_path.write_bytes(b"corrupt")
    with pytest.raises(WorldPackCorruptError, match="checksum mismatch"):
        _verify_replacement_world(_recipe(), manifest, tmp_path)


def test_policy_space_preflight_rejects_route_mode_change(tmp_path: Path):
    checkpoint = tmp_path / "source" / "checkpoints" / "iter_000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text("{}")
    (checkpoint.parent.parent / "experiment.yaml").write_text(yaml.safe_dump({
        "experiment": {"name": "space-test"},
        "env": {"agent_count": 1, "observation": {"mode": "box", "box_radius": 2},
                "geometry": {"grid_size": 32}},
        "training": {"algorithm": "ppo"},
    }))
    recipe = clone(checkpoint, "evaluation")
    from theseo_anysearch.experiments.execution_recipe import _migrate_config

    source, _ = _migrate_config(yaml.safe_load((checkpoint.parent.parent / "experiment.yaml").read_text()))
    changed = source.env.to_runtime_dict()
    changed["waypoint_curriculum"] = {
        "enabled": True, "initial_start": [1, 1, 1], "initial_goal": [3, 1, 1],
    }
    with pytest.raises(ValueError, match="observation space"):
        _verify_policy_spaces(recipe, changed, tmp_path, changed=True)


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


def test_interrupted_evaluation_records_failure_without_success_marker(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "source" / "checkpoints" / "iter_000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text("{}")
    (checkpoint.parent.parent / "experiment.yaml").write_text(yaml.safe_dump({
        "experiment": {"name": "interrupted"},
        "env": {"agent_count": 1, "observation": {"mode": "box", "box_radius": 1},
                "geometry": {"grid_size": 32}},
        "training": {"algorithm": "ppo"},
    }))

    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def numpy(self): return np.array([1.0], dtype=np.float32)

    class Algorithm:
        env_runner = SimpleNamespace(module=SimpleNamespace(state_dict=lambda: {"w": Tensor()}))
        stopped = False

        def restore(self, path): pass
        def stop(self): self.stopped = True

    algorithm = Algorithm()
    monkeypatch.setattr("theseo_anysearch.rllib.algorithms.ppo.PPOTrainer.build_algorithm_from_settings",
                        lambda settings, env_config: algorithm)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("stopped")
    monkeypatch.setattr("theseo_anysearch.experiments.execution_evaluate.collect_eval_episodes", interrupted)
    with pytest.raises(KeyboardInterrupt, match="stopped"):
        evaluate(clone(checkpoint, "evaluation"), base=tmp_path,
                 output_dir=tmp_path / "results", episodes=2, seed=1)
    run = next((tmp_path / "results").iterdir())
    assert algorithm.stopped
    assert not (run / "execution.json").exists()
    assert (run / "execution_failure.json").exists()
