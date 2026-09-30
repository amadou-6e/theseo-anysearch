"""Production compressed trajectory compatibility and publication tests."""
import json
import subprocess
import sys

import pytest
import zstandard

from theseo_anysearch.experiments.trajectory_storage import (
    find_trajectory, list_trajectories, read_trajectory, trajectory_stem, write_trajectory,
)


@pytest.fixture
def document():
    return {"schema_version": 2, "run_id": "test", "unknown": "é",
            "world": {"manifest_path": "../worlds/id/manifest.json", "extent": [80000, 2048, 512]},
            "episode": {"steps": [{"cursor_x": 70001, "reward": 0.12345678901234567,
                                   "actions": [0, 2], "cursors": [[70001, 3, 4], [9, 8, 7]],
                                   "mutations": [{"coordinate": [70001, 3, 4], "occupied": False,
                                                  "reward_weight": 0.375}]}]}}


def test_lossless_and_legacy(tmp_path, document):
    compressed = write_trajectory(tmp_path / "iter_000001.json.zst", document)
    legacy = tmp_path / "iter_000002.json"
    legacy.write_text(json.dumps(document), encoding="utf-8")
    assert read_trajectory(compressed) == read_trajectory(legacy) == document
    assert b'"world"' in zstandard.ZstdDecompressor().decompress(compressed.read_bytes())
    assert trajectory_stem(compressed) == "iter_000001"


def test_selection_deduplicates_and_excludes_sidecars(tmp_path, document):
    for name in ["iter_000002.json", "iter_000001.json", "best.json", "best_meta.json"]:
        (tmp_path / name).write_text("{}")
    for stem in ["iter_000002", "best"]:
        write_trajectory(tmp_path / (stem + ".json.zst"), document)
    (tmp_path / ".best.json.zst.tmp").write_text("partial")
    assert [p.name for p in list_trajectories(tmp_path)] == ["best.json.zst", "iter_000001.json", "iter_000002.json.zst"]
    assert find_trajectory(tmp_path, "best") == tmp_path / "best.json.zst"
    assert find_trajectory(tmp_path, "iter_000001") == tmp_path / "iter_000001.json"
    with pytest.raises(FileNotFoundError):
        find_trajectory(tmp_path, "iter_999999")


def test_atomic_failure_keeps_old_frame_and_cleans_temporary(tmp_path, document, monkeypatch):
    path = write_trajectory(tmp_path / "best.json.zst", document)
    previous = path.read_bytes()
    def fail(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr("theseo_anysearch.experiments.trajectory_storage.os.replace", fail)
    with pytest.raises(OSError):
        write_trajectory(path, {"episode": {"steps": []}})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".best.json.zst.*.tmp"))


@pytest.mark.parametrize("payload", [b"not-zstd", b"\x28\xb5\x2f\xfd", b"{}"])
def test_corrupt_compressed_file_is_not_legacy_fallback(tmp_path, payload):
    (tmp_path / "best.json").write_text("{}")
    (tmp_path / "best.json.zst").write_bytes(payload)
    with pytest.raises(zstandard.ZstdError):
        read_trajectory(find_trajectory(tmp_path, "best"))


def test_truncated_frame_and_nonfinite_value(tmp_path, document):
    path = write_trajectory(tmp_path / "best.json.zst", document)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(zstandard.ZstdError):
        read_trajectory(path)
    with pytest.raises(ValueError):
        write_trajectory(path, {"reward": float("nan")})
    with pytest.raises(ValueError):
        write_trajectory(tmp_path / "wrong.json", document)
    (tmp_path / "array.json").write_text("[]")
    with pytest.raises(ValueError):
        read_trajectory(tmp_path / "array.json")


def test_inspect_export_and_no_overwrite(tmp_path, document):
    path = write_trajectory(tmp_path / "best.json.zst", document)
    output = tmp_path / "export.json"
    command = [sys.executable, "-m", "theseo_anysearch.experiments.trajectory_storage", str(path), "--output", str(output)]
    subprocess.run(command, check=True)
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert subprocess.run(command, capture_output=True).returncode != 0
    selected = subprocess.run(command[:-2] + ["--step", "0"], capture_output=True, check=True)
    assert json.loads(selected.stdout) == document["episode"]["steps"][0]


def test_replay_cli_mixed_iterations(tmp_path, document):
    from theseo_anysearch.cli.commands.replay import _all_iter_trajectories, _find_trajectory
    directory = tmp_path / "trajectories"
    directory.mkdir()
    (directory / "iter_000001.json").write_text("{}")
    new = write_trajectory(directory / "iter_000002.json.zst", document)
    assert _find_trajectory(tmp_path, 2) == new
    assert len(_all_iter_trajectories(tmp_path)) == 2
    with pytest.raises(FileNotFoundError, match="Available"):
        _find_trajectory(tmp_path, 3)


def test_explain_selectors_support_new_and_legacy(tmp_path, document):
    from theseo_anysearch.rllib.explain.service import resolve_trajectory
    directory = tmp_path / "trajectories"
    directory.mkdir()
    (directory / "iter_000001.json").write_text("{}")
    new = write_trajectory(directory / "iter_000002.json.zst", document)
    best = write_trajectory(directory / "best.json.zst", document)
    assert resolve_trajectory(tmp_path, "latest") == new.resolve()
    assert resolve_trajectory(tmp_path, "best") == best.resolve()
    assert resolve_trajectory(tmp_path, "iter_000001").name == "iter_000001.json"
    assert resolve_trajectory(tmp_path, "iter_000002.json.zst") == new.resolve()


@pytest.mark.parametrize("nested", [False, True])
def test_garden_collection_reads_compressed_observations(tmp_path, nested):
    import numpy as np
    from theseo_anysearch.garden.collect import TrainingRunCollector
    from theseo_anysearch.garden.data_config import CacheConfig, TrainingRunSourceConfig
    directory = tmp_path / "trial" / "trajectories" if nested else tmp_path / "trajectories"
    document = {"episode": {"steps": [{"local_grid": [1] * 27}]}}
    write_trajectory(directory / "iter_000001.json.zst", document)
    (directory / "iter_000001.json").write_text(json.dumps(document))
    collector = TrainingRunCollector(TrainingRunSourceConfig(type="training_run", run="test:run"),
                                    CacheConfig(path=tmp_path / "cache"), box_radius=1)
    values = collector._extract_from_trajectories(tmp_path, None)
    assert values.shape == (1, 3, 3, 3)
    assert np.all(values == 1)
