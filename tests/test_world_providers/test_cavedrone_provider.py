"""CaveDrone wheel boundary and pinned-source end-to-end checks."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from theseo_anysearch.environments.cave_drone_export import (
    EXPECTED_EXTENT,
    GENERATOR_FAMILY,
    UPSTREAM_COMMIT,
)
from theseo_anysearch.environments.routing_manifests import RightsRecord
from theseo_anysearch.world_providers import service
from theseo_anysearch.world_providers.api import ProviderInfo
from theseo_anysearch.world_providers.api import ProviderParameter
from theseo_anysearch.world_providers.fixtures import FixtureBoxesProvider
from theseo_anysearch.world_providers.selection import add_to_experiment
from theseo_anysearch.experiments.loader import load_experiment


@pytest.fixture
def provider_module(monkeypatch: pytest.MonkeyPatch):
    package_root = Path(__file__).resolve().parents[2] / "providers" / "cavedrone"
    monkeypatch.syspath_prepend(str(package_root))
    return importlib.import_module("anysearch_cavedrone_provider")


def test_cavedrone_metadata_and_native_extent(provider_module) -> None:
    info = provider_module.Provider.info
    assert info.native_extent == EXPECTED_EXTENT
    assert info.native_meters_per_voxel == 0.5
    assert {item.name for item in info.parameters} == {"partition", "body-radius-m"}
    with pytest.raises(ValueError, match="native extent"):
        ProviderInfo("bad", "1", "bad", 0.5, native_extent=(192, 0, 192))
    parameter = ProviderParameter("style", "text", default="plain")
    assert ProviderInfo("legacy", "1", "legacy", 0.5, (parameter,)).parameters == (parameter,)


def test_world_selection_import_does_not_eagerly_load_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import theseo_anysearch.experiments.loader; "
            "assert 'torch' not in sys.modules"
        )],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_declared_native_extent_is_checked_before_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MisdeclaredFixture(FixtureBoxesProvider):
        info = replace(FixtureBoxesProvider.info, native_extent=EXPECTED_EXTENT)

    monkeypatch.setattr(service, "load_provider", lambda name: MisdeclaredFixture())
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    root = tmp_path / "world"
    with pytest.raises(ValueError, match="native extent"):
        service.generate_world("fixture-boxes", seed=1, output=root)
    assert not (root / "verification.json").exists()
    assert not (tmp_path / "registry.json").exists()


def test_generation_checks_source_rights_for_declared_split_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = service.load_bundle

    def evaluation_only(root: Path):
        bundle = original_load(root)
        rights = RightsRecord(
            status="reviewed", license_expression="MIT",
            allowed_uses=("evaluation",), evidence="test rights boundary",
        )
        return replace(bundle, source=bundle.source.model_copy(update={"rights": rights}))

    monkeypatch.setattr(service, "load_bundle", evaluation_only)
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    root = tmp_path / "world"
    with pytest.raises(PermissionError, match="training"):
        service.generate_world("fixture-boxes", seed=1, output=root)
    assert not (root / "verification.json").exists()
    assert not (tmp_path / "registry.json").exists()


def test_cavedrone_requires_explicit_source_and_valid_parameters(
    provider_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = provider_module.Provider()
    monkeypatch.delenv("ANYSEARCH_CAVEDRONE_SOURCE", raising=False)
    with pytest.raises(ValueError, match="ANYSEARCH_CAVEDRONE_SOURCE"):
        provider.generate(seed=42, output=tmp_path / "world", parameters={})
    with pytest.raises(ValueError, match="partition"):
        provider.generate(seed=42, output=tmp_path / "world", parameters={"partition": "bogus"})
    # See #493: `calibration` is a deprecated alias for `validation`; generation
    # must reject it, not silently accept it as an unrecognized legacy spelling.
    with pytest.raises(ValueError, match="partition"):
        provider.generate(seed=42, output=tmp_path / "world", parameters={"partition": "calibration"})
    with pytest.raises(ValueError, match="body-radius-m"):
        provider.generate(seed=42, output=tmp_path / "world", parameters={"body-radius-m": -0.1})
    with pytest.raises(ValueError, match="uint32"):
        provider.generate(seed=-1, output=tmp_path / "world", parameters={})
    assert not (tmp_path / "world").exists()


def test_rejected_task_set_is_never_registered_as_verified(
    provider_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = provider_module.Provider()
    monkeypatch.setenv("ANYSEARCH_CAVEDRONE_SOURCE", str(tmp_path / "source"))
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setattr(service, "load_provider", lambda name: provider)

    def no_tasks(source: Path, output: Path, **kwargs: object) -> dict:
        FixtureBoxesProvider().generate(seed=42, output=output, parameters={})
        (output / "task-00.json").unlink()
        (output / "reference-00.json").unlink()
        return {
            "upstream_commit": UPSTREAM_COMMIT,
            "topology_family": GENERATOR_FAMILY,
            "rejected_task_strata": ["source_start_fails_clearance"],
        }

    monkeypatch.setattr(provider_module, "export_seed", no_tasks)
    output = tmp_path / "world"
    with pytest.raises(ValueError, match="nonempty tasks"):
        service.generate_world("cavedrone", seed=42, output=output)
    assert not (output / "verification.json").exists()
    assert not (tmp_path / "registry.json").exists()


def test_explicit_shared_study_id_catches_train_test_root_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    root = tmp_path / "world"
    service.generate_world("fixture-boxes", seed=42, output=root)
    train = tmp_path / "train.yaml"
    train.write_text(
        "experiment:\n  name: shared-study-test\nenv:\n  agent_count: 1\n"
        "  geometry:\n    grid_size: 12\ntraining:\n  algorithm: ppo\n"
        "  runner: local\nworlds:\n  role: train\n"
        "  study_id: one-study\n  study_root: .\n",
        encoding="utf-8",
    )
    add_to_experiment(root, train)
    test = tmp_path / "test.yaml"
    test.write_text(
        train.read_text(encoding="utf-8").replace("role: train", "role: test"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="crosses split roles"):
        load_experiment(train)


def test_unrelated_yaml_sequence_does_not_block_study_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    root = tmp_path / "world"
    service.generate_world("fixture-boxes", seed=42, output=root)
    (tmp_path / "notes.yaml").write_text("- unrelated\n- list\n", encoding="utf-8")
    train = tmp_path / "train.yaml"
    train.write_text(
        "experiment:\n  name: study-scan-test\nenv:\n  agent_count: 1\n"
        "  geometry:\n    grid_size: 12\ntraining:\n  algorithm: ppo\n"
        "  runner: local\nworlds:\n  role: train\n"
        "  study_id: one-study\n  study_root: .\n",
        encoding="utf-8",
    )
    assert add_to_experiment(root, train)["status"] == "added"


def test_pinned_source_repeat_pngs_and_yaml_round_trip(
    provider_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = os.getenv("ANYSEARCH_CAVEDRONE_SOURCE")
    if not source_path:
        pytest.skip("explicit pinned CaveDroneSim checkout not configured")
    provider = provider_module.Provider()
    monkeypatch.setattr(service, "load_provider", lambda name: provider)
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    first = tmp_path / "first"
    second = tmp_path / "second"
    report_a = service.generate_world("cavedrone", seed=42, output=first)
    report_b = service.generate_world("cavedrone", seed=42, output=second)
    assert report_a["files"] == report_b["files"]
    assert report_a["previews"] == report_b["previews"]
    assert report_a["world_identity_sha256"] == report_b["world_identity_sha256"]
    assert report_a["topology_family"] == GENERATOR_FAMILY
    assert len(list(first.glob("task-*.json"))) > 0
    for name in report_a["previews"]:
        with Image.open(first / name) as preview:
            preview.verify()
    bundle = service.load_verified_world(first, use="training")
    assert bundle.world.extent.as_tuple() == EXPECTED_EXTENT
    assert bundle.world.frame.meters_per_voxel == 0.5

    config = tmp_path / "train.yaml"
    config.write_text(
        "experiment:\n  name: cavedrone-provider-test\nenv:\n  agent_count: 1\n"
        "  geometry:\n    grid_size: 192\ntraining:\n  algorithm: ppo\n"
        "  runner: local\nworlds:\n  role: train\n"
        "  study_id: cavedrone-provider-test\n  study_root: .\n",
        encoding="utf-8",
    )
    result = add_to_experiment(first, config)
    assert result["status"] == "added"
    assert add_to_experiment(first, config)["status"] == "unchanged"
