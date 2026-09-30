import hashlib
import io
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "providers/gazebo/src"))
from anysearch_gazebo import Provider
from theseo_anysearch.world_providers.source_cache import cached_sources


@pytest.mark.parametrize("parameters", [{"layout": "unknown"}, {"meters-per-voxel": 0.1},
    {"meters-per-voxel": 0.3}, {"body-radius-m": float("nan")}, {"unknown": 1}])
def test_invalid_parameters_do_not_download(tmp_path, parameters):
    with patch("anysearch_gazebo.cached_sources") as download:
        with pytest.raises(ValueError):
            Provider().generate(seed=42, output=tmp_path / "world", parameters=parameters)
        download.assert_not_called()


def test_extra_and_entrypoint():
    import tomllib
    root = Path(__file__).resolve().parents[2]
    core = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    optional = tomllib.loads((root / "providers/gazebo/pyproject.toml").read_text())["project"]
    assert core["optional-dependencies"]["gazebo"] == ["theseo-anysearch-gazebo>=0.1.0,<0.2.0"]
    assert optional["entry-points"]["theseo_anysearch.world_providers"]["gazebo"] == "anysearch_gazebo:Provider"
    assert not any("theseo-anysearch-gazebo" in dependency for dependency in core["dependencies"])


def test_cli_help_does_not_download():
    # Invokes the real root app object as a genuine subprocess (rather than
    # typer.testing.CliRunner, which invokes commands directly and would not
    # surface a mismatch between the dynamic provider command's exception
    # family and the root app's error handler — see #471).
    import subprocess
    import sys
    import textwrap

    provider_src = str(Path(Path(__file__).resolve().parents[2], "providers", "gazebo", "src"))
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {provider_src!r})
        from unittest.mock import patch
        from anysearch_gazebo import Provider

        with patch("theseo_anysearch.cli.commands.worlds.load_provider", return_value=Provider()), \\
             patch("anysearch_gazebo.cached_sources") as download:
            from theseo_anysearch.cli.main import app
            try:
                app(["worlds", "gazebo", "--help"])
            except SystemExit as exc:
                assert exc.code in (0, None), exc.code
            assert not download.called
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    help_text = " ".join(result.stdout.split())
    assert "roofed" in help_text
    assert "meters-per-voxel" in help_text


def test_large_archive_cache_download_and_offline_reuse(tmp_path):
    data = b"a" * (2 * 1024 * 1024 + 1)
    hashes = {"common_models.tar.xz": hashlib.sha256(data).hexdigest()}
    response = Mock()
    response.read.return_value = data
    response.geturl.return_value = "https://example.org/common_models.tar.xz"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch("theseo_anysearch.world_providers.source_cache.urlopen", return_value=response) as download:
        root = cached_sources(tmp_path, base_url="https://example.org", hashes=hashes, max_file_bytes=4 * 1024 * 1024)
        assert cached_sources(tmp_path, base_url="https://example.org", hashes=hashes,
            offline=True, max_file_bytes=4 * 1024 * 1024) == root
        assert download.call_count == 1
        with pytest.raises(ValueError, match="hash mismatch"):
            cached_sources(tmp_path, base_url="https://example.org", hashes=hashes)
    (root / "common_models.tar.xz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        cached_sources(tmp_path, base_url="https://example.org", hashes=hashes, offline=True, max_file_bytes=4 * 1024 * 1024)


@pytest.mark.parametrize("limit", [0, -1, 64 * 1024 * 1024, True])
def test_invalid_cache_limit(tmp_path, limit):
    with pytest.raises(ValueError):
        cached_sources(tmp_path, base_url="https://example.org", hashes={"a": "0" * 64}, max_file_bytes=limit)


def test_archive_decompression_and_duplicate_limits(tmp_path):
    from theseo_anysearch.environments.gazebo_maze_export import SdfArchives
    for duplicate in (False, True):
        with tarfile.open(tmp_path / "3d_maze.tar.xz", "w:xz") as archive:
            if duplicate:
                for _ in range(2):
                    archive.addfile(tarfile.TarInfo("same"))
            else:
                member = tarfile.TarInfo("huge")
                member.size = 2 * 1024 * 1024 + 1
                archive.addfile(member, io.BytesIO(b"a" * member.size))
        with pytest.raises(ValueError):
            SdfArchives(tmp_path, verify_hashes=False)


def test_generation_pipeline_with_collision_fixture(tmp_path, monkeypatch):
    import numpy as np
    from theseo_anysearch.environments.gazebo_maze_export import Collision
    from theseo_anysearch.world_providers.bundle import load_bundle
    from theseo_anysearch.world_providers.service import generate_world, load_verified_world
    source = tmp_path / "source"
    source.mkdir()
    (source / "LICENSE").write_text("independent test-fixture license")
    wall = Collision("fixture-wall", "box", (0, 0, 4), np.eye(3), size_m=(0.8, 10, 8))
    ground = Collision("fixture-ground", "box", (0, 0, -0.05), np.eye(3), size_m=(92, 92, 0.1))
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    bundles = []
    with patch("anysearch_gazebo.cached_sources", return_value=source), \
         patch("theseo_anysearch.environments.gazebo_maze_export.read_source_collisions", return_value=((wall, ground), {})), \
         patch("theseo_anysearch.world_providers.service.load_provider", return_value=Provider()):
        for resolution in (0.5, 0.25):
            output = tmp_path / str(resolution)
            report = generate_world("gazebo", seed=42, output=output, parameters={"meters-per-voxel": resolution})
            assert len(report["previews"]) == 6
            bundles.append(load_verified_world(output, use="evaluation"))
            with pytest.raises(PermissionError, match="training is not cleared"):
                load_bundle(output, use="training")
            from theseo_anysearch.world_providers.selection import add_to_experiment
            config = tmp_path / "train.yaml"
            original = "env:\n  geometry: {}\ntraining: {}\n"
            config.write_text(original)
            with pytest.raises(PermissionError, match="training is not cleared"):
                add_to_experiment(output, config)
            assert config.read_text() == original
    assert bundles[0].world.root_geometry_id == bundles[1].world.root_geometry_id
    assert bundles[0].world.identity_sha256 != bundles[1].world.identity_sha256
    assert bundles[0].world.extent.as_tuple() == (184, 18, 184)
    assert bundles[1].world.extent.as_tuple() == (368, 36, 368)
    assert bundles[0].world.frame.storage_axes_in_source == (
        (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
    )
    assert bundles[0].conversion.parameters["original_open_top_world_sha256"]
    assert (bundles[0].root / "world-open-top.json").exists()


def test_zero_tasks_never_published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with patch("anysearch_gazebo.cached_sources", return_value=source), \
         patch("anysearch_gazebo.export_maze", return_value={"accepted_queries": []}):
        with pytest.raises(ValueError, match="no accepted"):
            Provider().generate(seed=42, output=tmp_path / "world", parameters={})
    assert not (tmp_path / "world").exists()
