import json
import hashlib
import platform
import sys
from pathlib import Path
import pytest
import yaml

from theseo_anysearch.experiments.execution_recipe import (
    ExecutionRecipe,
    GeometryDecision,
    clone,
    make_portable,
    validate,
)


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
    assert result["execution_supported"] is True
    assert loaded.policy_contract["action"]["mode"] == "discrete_18"


def test_world_difference_and_tamper_detection(tmp_path):
    checkpoint = fixture(tmp_path)
    config_path = checkpoint.parent.parent / "experiment.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["env"]["waypoint_curriculum"] = {
        "enabled": True, "initial_start": [1, 1, 1], "initial_goal": [2, 1, 1]
    }
    config_path.write_text(yaml.safe_dump(config))
    recipe = clone(checkpoint, "continuation")
    world = tmp_path / "manifest.json"
    world.write_text(json.dumps({
        "schema_version": 1, "coordinate_type": "u32",
        "storage_coordinate_convention": "zero_based",
        "environment_coordinate_convention": "one_based",
        "environment_min": [1, 1, 1], "source_origin": [0, 0, 0],
        "extent": {"x": 64, "y": 32, "z": 32},
        "chunk_shape": {"x": 32, "y": 32, "z": 32}, "chunks": [],
        "identity_sha256": "1" * 64,
    }))
    recipe.overrides.task = GeometryDecision.preserve()
    recipe.overrides.routes = GeometryDecision.preserve()
    result = validate(recipe, world)
    assert [change["component"] for change in result["changes"]] == ["world.extent", "world"]
    Path(recipe.experiment.path).write_text("tampered")
    with pytest.raises(ValueError, match="tampered"):
        validate(recipe)


def test_clone_explicitly_migrates_legacy_config_and_materializes_defaults(tmp_path):
    checkpoint = fixture(tmp_path)
    path = checkpoint.parent.parent / "experiment.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["env"] = {"agent_count": 1, "obs_mode": "box", "box_radius": 1,
                  "action_mode": "discrete_18", "grid_size": 32}
    path.write_text(yaml.safe_dump(raw))
    recipe = clone(checkpoint, "evaluation")
    assert "env.obs_mode -> env.observation.mode" in recipe.config_migration.transforms
    assert "env.max_steps" in recipe.config_migration.materialized_defaults
    assert recipe.schema_version == 3
    assert recipe.policy_contract["connectors"]["api_stack"] == "connector_v2"


def test_policy_contract_drift_is_rejected_even_with_updated_artifact_hash(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    path = Path(recipe.experiment.path)
    raw = yaml.safe_load(path.read_text())
    raw["env"]["action"]["mode"] = "discrete_6"
    path.write_text(yaml.safe_dump(raw))
    from theseo_anysearch.experiments.execution_recipe import _sha
    recipe.experiment.sha256 = _sha(path)
    with pytest.raises(ValueError, match="migration record|policy contract"):
        validate(recipe)


def test_schema_one_recipe_is_upgraded_from_its_archived_config(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    payload = recipe.model_dump(mode="json")
    payload["schema_version"] = 1
    payload.pop("config_migration")
    payload["policy_contract"] = {"legacy": True}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(payload))
    loaded = ExecutionRecipe.load(path)
    assert loaded.schema_version == 3
    assert loaded.provenance["recipe_migrations"] == ["1 -> 2", "2 -> 3"]
    assert loaded.policy_contract["action"]["mode"] == "discrete_18"
    assert validate(loaded, base=tmp_path)["execution_supported"]


def test_historical_recipe_cannot_read_outside_bundle_during_migration(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    payload = recipe.model_dump(mode="json")
    payload["schema_version"] = 1
    payload["experiment"]["path"] = "../run/experiment.yaml"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    path = bundle / "recipe.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="escapes the recipe bundle"):
        ExecutionRecipe.load(path)


def test_disabled_capability_requires_explicit_replacement(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    recipe.extension_bindings = ["reward:segment_countdown_goal"]
    recipe.overrides.disabled_capabilities = ["reward:segment_countdown_goal"]
    with pytest.raises(ValueError, match="explicit replacement"):
        validate(recipe)
    recipe.overrides.replacements["reward:segment_countdown_goal"] = "reward:builtin"
    with pytest.raises(ValueError, match="not selected"):
        validate(recipe)
    recipe.overrides.disabled_capabilities = ["reward:unknown"]
    recipe.overrides.replacements = {"reward:unknown": "reward:builtin"}
    with pytest.raises(ValueError, match="does not exist"):
        validate(recipe)


def test_scenario_replacement_covers_evaluation_only_selection(tmp_path):
    checkpoint = fixture(tmp_path)
    path = checkpoint.parent.parent / "experiment.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["evaluation"] = {"scenarios": {"provider": {"name": "seeded_scenario"}}}
    path.write_text(yaml.safe_dump(raw))
    recipe = clone(checkpoint, "evaluation")
    recipe.extension_bindings = ["scenario:seeded_scenario"]
    recipe.overrides.disabled_capabilities = ["scenario:seeded_scenario"]
    recipe.overrides.replacements = {"scenario:seeded_scenario": "scenario:none"}
    result = validate(recipe)
    assert result["effective_config"]["evaluation"]["scenarios"]["provider"] is None


def test_world_swap_requires_explicit_geometry_dependent_decisions(tmp_path):
    checkpoint = fixture(tmp_path)
    config_path = checkpoint.parent.parent / "experiment.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["env"]["waypoints_file"] = "old-waypoints.json"
    raw["env"]["geometry"]["boxes"] = [[1, 1, 1, 2, 2, 2]]
    raw["env"]["waypoint_curriculum"] = {
        "enabled": True, "initial_start": [1, 1, 1], "initial_goal": [2, 1, 1]
    }
    config_path.write_text(yaml.safe_dump(raw))
    recipe = clone(checkpoint, "evaluation")
    world = tmp_path / "manifest.json"
    world.write_text(json.dumps({
        "schema_version": 1, "coordinate_type": "u32",
        "storage_coordinate_convention": "zero_based",
        "environment_coordinate_convention": "one_based",
        "environment_min": [1, 1, 1], "source_origin": [0, 0, 0],
        "extent": {"x": 64, "y": 32, "z": 32},
        "chunk_shape": {"x": 32, "y": 32, "z": 32}, "chunks": [],
        "identity_sha256": "2" * 64,
    }))
    with pytest.raises(ValueError, match="explicit task and routes"):
        validate(recipe, world)
    recipe.overrides.task = GeometryDecision(mode="clear")
    recipe.overrides.routes = GeometryDecision(mode="clear")
    with pytest.raises(ValueError, match="no waypoint, route, curriculum, or scenario"):
        validate(recipe, world)
    recipe.overrides.routes = GeometryDecision.preserve()
    result = validate(recipe, world)
    assert result["effective_config"]["env"]["waypoint_curriculum"]["enabled"] is True
    assert result["effective_config"]["env"]["geometry"]["boxes"] is None
    assert result["changes"][-1]["to"] == "2" * 64


def test_task_decision_without_world_is_rejected(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    recipe.overrides.task = GeometryDecision(mode="clear")
    with pytest.raises(ValueError, match="require a replacement world"):
        validate(recipe)


def test_no_reward_replacement_is_explicit_and_preserves_other_capabilities(tmp_path):
    checkpoint = fixture(tmp_path)
    config_path = checkpoint.parent.parent / "experiment.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["env"]["rewards"] = {"provider": "shaped", "goal_reward": 7.0,
                              "step_cost": -1.0, "distance_reward_mode": "zone"}
    config_path.write_text(yaml.safe_dump(raw))
    recipe = clone(checkpoint, "evaluation")
    recipe.extension_bindings = ["reward:shaped", "scenario:kept"]
    recipe.overrides.disabled_capabilities = ["reward:shaped"]
    recipe.overrides.replacements = {"reward:shaped": "reward:none"}
    result = validate(recipe)
    rewards = result["effective_config"]["env"]["rewards"]
    assert rewards["provider"] is None
    assert rewards["goal_reward"] == rewards["step_cost"] == 0.0
    assert rewards["distance_reward_mode"] == "progress"
    assert result["active_extension_bindings"] == ["scenario:kept"]


def test_capability_replacement_must_be_qualified_and_kind_compatible(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    recipe.extension_bindings = ["reward:shaped"]
    recipe.overrides.disabled_capabilities = ["reward:shaped"]
    recipe.overrides.replacements = {"reward:shaped": "scenario:none"}
    with pytest.raises(ValueError, match="changes kind"):
        validate(recipe)


def test_selected_action_capability_is_replaced_in_effective_pipeline(tmp_path):
    checkpoint = fixture(tmp_path)
    config_path = checkpoint.parent.parent / "experiment.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["env"]["action"]["predicates"] = ["custom_gate", "bounds"]
    config_path.write_text(yaml.safe_dump(raw))
    recipe = clone(checkpoint, "evaluation")
    recipe.extension_bindings = ["predicate:custom_gate", "reward:kept"]
    recipe.overrides.disabled_capabilities = ["predicate:custom_gate"]
    recipe.overrides.replacements = {"predicate:custom_gate": "predicate:valid_action"}
    result = validate(recipe)
    assert [item["name"] for item in result["effective_config"]["env"]["action"]["predicates"]] == [
        "valid_action", "bounds"
    ]
    assert result["active_extension_bindings"] == ["reward:kept"]


def test_multiple_capabilities_are_replaced_without_disabling_unrelated_binding(tmp_path):
    checkpoint = fixture(tmp_path)
    path = checkpoint.parent.parent / "experiment.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["env"]["rewards"] = {"provider": "shaped"}
    raw["env"]["action"]["predicates"] = ["custom_gate", "bounds"]
    path.write_text(yaml.safe_dump(raw))
    recipe = clone(checkpoint, "evaluation")
    recipe.extension_bindings = ["reward:shaped", "predicate:custom_gate", "scenario:kept"]
    recipe.overrides.disabled_capabilities = ["reward:shaped", "predicate:custom_gate"]
    recipe.overrides.replacements = {
        "reward:shaped": "reward:builtin",
        "predicate:custom_gate": "predicate:valid_action",
    }
    report = validate(recipe)
    assert report["active_extension_bindings"] == ["scenario:kept"]
    assert report["effective_config"]["env"]["rewards"]["provider"] is None
    assert [item["name"] for item in report["effective_config"]["env"]["action"]["predicates"]] == [
        "valid_action", "bounds",
    ]


def test_multi_capability_manifest_accepts_builtin_action_pipeline(tmp_path):
    checkpoint = fixture(tmp_path)
    extension = checkpoint.parent.parent / "native_extension"
    extension.mkdir()
    binary = extension / "rules.dll"
    binary.write_bytes(b"test library")
    (extension / "extension.json").write_text(json.dumps({
        "abi_version": 2,
        "source_sha256": "0" * 64,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "library": binary.name,
        "capabilities": ["reward", "predicate"],
        "rewards": ["shaped"],
        "predicates": ["custom_gate"],
        "platform": sys.platform,
        "machine": platform.machine(),
    }))
    recipe = clone(checkpoint, "evaluation")
    assert validate(recipe)["valid"]


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


def test_relative_artifact_cannot_escape_portable_bundle(tmp_path):
    recipe = clone(fixture(tmp_path), "evaluation")
    bundle = tmp_path / "bundle"
    portable = make_portable(recipe, bundle)
    portable.experiment.path = "../run/experiment.yaml"
    with pytest.raises(ValueError, match="escapes the recipe bundle"):
        validate(portable, base=bundle)


def test_portable_bundle_preserves_extension_layout(tmp_path):
    checkpoint = fixture(tmp_path)
    extension = checkpoint.parent.parent / "native_extension"
    extension.mkdir()
    binary = extension / "rules.dll"
    binary.write_bytes(b"library")
    (extension / "extension.json").write_text(json.dumps({"library": binary.name,
        "capabilities": ["reward"], "rewards": ["shaped"], "predicates": [],
        "outcomes": [], "scenarios": [], "geometries": []}))
    portable = make_portable(clone(checkpoint, "evaluation"), tmp_path / "bundle")
    paths = {artifact.role: artifact.path for artifact in portable.extension}
    assert Path(paths["extension_manifest"]).parent == Path(paths["extension_binary"]).parent
    assert Path(paths["extension_binary"]).name == "rules.dll"


def test_source_revision_is_detected(tmp_path):
    checkpoint = fixture(tmp_path)
    (checkpoint.parent.parent / "provenance.json").write_text(json.dumps({"source_commit": "abc123"}))
    recipe = clone(checkpoint, "evaluation")
    assert recipe.provenance["source_revision"] == "abc123"
    assert recipe.provenance_gaps == []


def test_bundle_failure_is_clean_and_never_overwrites(tmp_path, monkeypatch):
    recipe = clone(fixture(tmp_path), "evaluation")
    bundle = tmp_path / "bundle"
    monkeypatch.setattr("theseo_anysearch.experiments.execution_recipe.shutil.copytree",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy failed")))
    with pytest.raises(OSError, match="copy failed"):
        make_portable(recipe, bundle)
    assert not bundle.exists()
    assert not list(tmp_path.glob(".bundle.*"))
    bundle.mkdir()
    with pytest.raises(FileExistsError):
        make_portable(recipe, bundle)
