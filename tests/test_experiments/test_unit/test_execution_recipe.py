import hashlib
import json
from pathlib import Path
import pytest
import yaml

from theseo_anysearch.experiments.execution_recipe import ExecutionRecipe, clone, make_portable, validate


def fixture(tmp_path: Path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "iter_000010"
    checkpoint.mkdir(parents=True)
    config = {
        "experiment": {"name": "x", "output_dir": str(tmp_path)},
        "env": {"agent_count": 1,
                "observation": {"mode": "box", "box_radius": 1},
                "action": {"mode": "discrete_18"},
                "geometry": {"grid_size": 32}},
        "training": {"algorithm": "ppo", "model": "voxel_encoder", "iterations": 10},
    }
    (run / "experiment.yaml").write_text(yaml.safe_dump(config))
    (checkpoint / "state.json").write_text(json.dumps({"iteration": 10, "rllib_path": str(checkpoint),
        "world_contract": {"extent": [32, 32, 32], "identity_sha256": "old"}}))
    return checkpoint


def test_clone_round_trip_and_scope(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    path = tmp_path / "recipe.yaml"
    recipe.save(path)
    loaded = ExecutionRecipe.load(path)
    assert loaded == recipe
    result = validate(loaded)
    assert result["valid"] and "optimizer" in result["inactive"]
    assert result["execution_supported"] is False
    assert loaded.policy_contract["action"]["mode"] == "discrete_18"


def test_world_difference_and_tamper_detection(tmp_path):
    recipe = clone(fixture(tmp_path), "continuation")
    world = tmp_path / "manifest.json"
    world.write_text(json.dumps({"extent": [64, 32, 32], "identity_sha256": "new"}))
    result = validate(recipe, world)
    assert [change["component"] for change in result["changes"]] == ["world.extent", "world"]
    Path(recipe.experiment.path).write_text("tampered")
    with pytest.raises(ValueError, match="tampered"):
        validate(recipe)


def test_disabled_capability_requires_explicit_replacement(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    recipe.extension_bindings = ["reward:segment_countdown_goal"]
    recipe.overrides.disabled_capabilities = ["reward:segment_countdown_goal"]
    with pytest.raises(ValueError, match="explicit replacements"):
        validate(recipe)
    recipe.overrides.replacements["reward:segment_countdown_goal"] = "reward:builtin"
    assert validate(recipe)["valid"]
    recipe.overrides.disabled_capabilities = ["reward:unknown"]
    with pytest.raises(ValueError, match="do not exist"):
        validate(recipe)


def test_missing_parent_provenance_fails(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "iter_1"
    checkpoint.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="experiment.yaml"):
        clone(checkpoint, "evaluation")


def test_portable_bundle_relocates_and_verifies(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    bundle = tmp_path / "bundle"
    portable = make_portable(recipe, bundle)
    portable.save(bundle / "recipe.yaml")
    moved = tmp_path / "moved"
    bundle.rename(moved)
    loaded = ExecutionRecipe.load(moved / "recipe.yaml")
    assert validate(loaded, base=moved)["valid"]
    assert not Path(loaded.checkpoint.path).is_absolute()
    (moved / loaded.experiment.path).write_text("tampered")
    with pytest.raises(ValueError, match="tampered"):
        validate(loaded, base=moved)


def test_source_revision_is_detected(tmp_path):
    checkpoint = fixture(tmp_path)
    (checkpoint.parent.parent / "provenance.json").write_text(json.dumps({"source_commit": "abc123"}))
    recipe = clone(checkpoint, "evaluation")
    assert recipe.provenance["source_revision"] == "abc123"
    assert recipe.provenance_gaps == []
