"""Offline conformance checks for installable voxel-world providers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from theseo_anysearch.cli.main import app
from theseo_anysearch.experiments.loader import load_experiment
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.world_providers.service import generate_world, load_verified_world
from theseo_anysearch.world_providers import api as provider_api
from theseo_anysearch.world_providers import service as provider_service


def test_world_cli_and_loader_do_not_require_optional_torch() -> None:
    code = """
import sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('optional Torch must not be imported')
sys.meta_path.insert(0, NoTorch())
from theseo_anysearch.cli.main import app
from theseo_anysearch.experiments.loader import load_experiment
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(autouse=True)
def isolate_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))


def _world(tmp_path: Path, seed: int = 42) -> Path:
    root = tmp_path / f"world-{seed}"
    generate_world("fixture-boxes", seed=seed, output=root)
    return root


def _config(path: Path) -> None:
    path.write_text(
        "experiment:\n  name: fixture-run\nenv:\n  agent_count: 1\n"
        "  geometry:\n    grid_size: 12\ntraining:\n  algorithm: ppo\n  runner: local\n",
        encoding="utf-8",
    )


def test_fixture_is_deterministic_and_has_checked_pngs(tmp_path: Path) -> None:
    first = _world(tmp_path, 7)
    second = tmp_path / "repeat"
    generate_world("fixture-boxes", seed=7, output=second)
    assert (first / "occupancy.npy").read_bytes() == (second / "occupancy.npy").read_bytes()
    assert load_verified_world(first, use="training").world.identity_sha256 == load_verified_world(second).world.identity_sha256
    assert sorted(path.name for path in (first / "previews").glob("*.png")) == [
        "xy-projection.png", "xy-slice.png", "xz-projection.png", "xz-slice.png",
        "yz-projection.png", "yz-slice.png",
    ]


def test_changed_occupancy_is_rejected(tmp_path: Path) -> None:
    root = _world(tmp_path)
    with (root / "occupancy.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="occupied voxel grid"):
        load_verified_world(root)


def test_changed_local_source_or_preview_is_rejected(tmp_path: Path) -> None:
    root = _world(tmp_path)
    (root / "generator.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact SHA|artifact SHA"):
        load_verified_world(root)
    root = tmp_path / "preview-world"
    generate_world("fixture-boxes", seed=42, output=root)
    with (root / "previews" / "xy-projection.png").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="PNG preview"):
        load_verified_world(root)


def test_preview_failure_never_marks_world_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_preview(*args, **kwargs):
        raise OSError("headless renderer failed")

    monkeypatch.setattr(provider_service, "render_previews", fail_preview)
    root = tmp_path / "preview-failure"
    with pytest.raises(OSError, match="headless renderer failed"):
        generate_world("fixture-boxes", seed=42, output=root)
    assert not (root / "verification.json").exists()
    assert not (tmp_path / "registry.json").exists()


def test_provider_parameter_bounds_and_remote_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    bad = runner.invoke(app, ["worlds", "fixture-boxes", "--seed", "1", "--output", str(tmp_path / "bad"), "--wall-height", "9"])
    assert bad.exit_code != 0
    assert not (tmp_path / "bad").exists()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "providers": [{
        "name": "sample", "distribution": "sample-worlds", "version": "1.0"
    }]}), encoding="utf-8")
    monkeypatch.setenv("ANYSEARCH_WORLDS_CATALOG", str(catalog))
    result = runner.invoke(app, ["worlds", "list", "--remote"])
    assert result.exit_code == 0
    assert "sample-worlds==1.0" in result.output


def test_broken_optional_provider_does_not_break_builtin_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEntryPoint:
        name = "broken-provider"

        def load(self):
            raise ImportError("optional native dependency missing")

    monkeypatch.setattr(provider_api, "entry_points", lambda *, group: [BrokenEntryPoint()])
    result = CliRunner().invoke(app, ["worlds", "list"])
    assert result.exit_code == 0, result.output
    assert "fixture-boxes" in result.output
    assert "broken-provider  unavailable" in result.output


def test_provider_parameter_schema_rejects_non_numeric_bounds() -> None:
    with pytest.raises(ValueError, match="only numeric"):
        provider_api.ProviderParameter("style", "text", minimum=0)


def test_cli_lists_and_generates(tmp_path: Path) -> None:
    runner = CliRunner()
    assert "fixture-boxes" in runner.invoke(app, ["worlds", "list"]).output
    root = tmp_path / "cli-world"
    result = runner.invoke(app, ["worlds", "fixture-boxes", "--seed", "2", "--output", str(root)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tasks_verified"] == 1
    assert (root / "verification.json").exists()
    assert runner.invoke(app, ["worlds", "fixture-boxes", "--seed", "2", "--output", str(root)]).exit_code != 0


def test_add_produces_loadable_config_and_is_idempotent(tmp_path: Path) -> None:
    root = _world(tmp_path)
    config = tmp_path / "experiment.yaml"
    _config(config)
    config.write_text(config.read_text(encoding="utf-8").replace("grid_size: 12", "grid_size: 12  # keep this note"), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["worlds", "add", str(root), "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "# keep this note" in config.read_text(encoding="utf-8")
    assert "+++" in result.output
    loaded = load_experiment(config)
    assert loaded.worlds is not None
    assert loaded.worlds.world_identity_sha256 == load_verified_world(root).world.identity_sha256
    assert loaded.env.geometry.compiled_world_path is not None
    runtime = VoxelEnv(loaded.env.to_runtime_dict())
    runtime.reset(seed=11)
    runtime.close()
    saved = config.read_bytes()
    result = runner.invoke(app, ["worlds", "add", str(root), "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "unchanged"
    assert config.read_bytes() == saved


def test_add_rejects_conflicting_geometry_and_split(tmp_path: Path) -> None:
    root = _world(tmp_path)
    config = tmp_path / "experiment.yaml"
    _config(config)
    config.write_text(config.read_text().replace("grid_size: 12", "grid_size: 12\n    pool:\n      pool_dir: other"))
    result = CliRunner().invoke(app, ["worlds", "add", str(root), "--config", str(config)])
    assert result.exit_code != 0
    assert "another geometry" in result.output
    config.write_text(
        _base_config_with_role("test"), encoding="utf-8"
    )
    result = CliRunner().invoke(app, ["worlds", "add", str(root), "--config", str(config)])
    assert result.exit_code != 0
    assert "training worlds only" in result.output


def test_loader_rechecks_world_after_manual_yaml_or_artifact_edit(tmp_path: Path) -> None:
    root = _world(tmp_path)
    config = tmp_path / "experiment.yaml"
    _config(config)
    assert CliRunner().invoke(app, ["worlds", "add", str(root), "--config", str(config)]).exit_code == 0
    raw = config.read_text(encoding="utf-8")
    config.write_text(raw.replace("role: train", "role: test"), encoding="utf-8")
    with pytest.raises((ValueError, PermissionError), match="split role|another split"):
        load_experiment(config)
    config.write_text(raw, encoding="utf-8")
    (root / "split.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment(config)


def test_same_study_second_yaml_cannot_reassign_root_to_test(tmp_path: Path) -> None:
    root = _world(tmp_path)
    train = tmp_path / "train.yaml"
    _config(train)
    assert CliRunner().invoke(app, ["worlds", "add", str(root), "--config", str(train)]).exit_code == 0
    test = tmp_path / "test.yaml"
    test.write_text(train.read_text(encoding="utf-8").replace("role: train", "role: test"), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses split roles"):
        load_experiment(train)


def test_same_study_rejects_different_split_record(tmp_path: Path) -> None:
    first = _world(tmp_path, 1)
    second = _world(tmp_path, 2)
    train = tmp_path / "train.yaml"
    _config(train)
    assert CliRunner().invoke(app, ["worlds", "add", str(first), "--config", str(train)]).exit_code == 0
    another = tmp_path / "another.yaml"
    _config(another)
    another.write_text(another.read_text(encoding="utf-8") + "worlds:\n  role: train\n  study_id: train\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["worlds", "add", str(second), "--config", str(another)])
    assert result.exit_code != 0
    assert "same-study YAMLs must reference one split record" in result.output


def _base_config_with_role(role: str) -> str:
    return (
        "experiment:\n  name: fixture-run\nenv:\n  agent_count: 1\n"
        "training:\n  algorithm: ppo\nworlds:\n  role: " + role + "\n"
    )
