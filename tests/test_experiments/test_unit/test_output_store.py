"""Unit tests for experiment output storage."""

from __future__ import annotations

from pathlib import Path

from theseo_anysearch.experiments.output import OutputStore


class TestOutputStore:
    """Verify JSON and directory behaviors of the output store."""

    def test_write_read_json(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        store.write_json("meta.json", {"key": "value"})
        assert store.read_json("meta.json") == {"key": "value"}

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        store.write_json("nested/deep/file.json", {})
        assert tmp_path.joinpath("run", "nested", "deep", "file.json").exists()

    def test_exists_true(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        store.write_json("x.json", {})
        assert store.exists("x.json")

    def test_exists_false(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        assert not store.exists("missing.json")

    def test_list_returns_relative_paths(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        store.write_json("a/b.json", {})
        store.write_json("c.json", {})
        paths = store.list()
        assert "a/b.json" in paths
        assert "c.json" in paths

    def test_list_dirs(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        tmp_path.joinpath("run", "checkpoints", "iter_000001").mkdir(parents=True)
        tmp_path.joinpath("run", "checkpoints", "iter_000002").mkdir(parents=True)
        dirs = store.list_dirs("checkpoints")
        assert any("iter_000001" in entry for entry in dirs)
        assert any("iter_000002" in entry for entry in dirs)


class TestOutputStoreExtra:
    """Verify bytes and YAML helpers on the output store."""

    def test_write_read_bytes(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        data = b"\x00\x01\x02proto"
        store.write_bytes("traj/episode_001.pb", data)
        assert store.read_bytes("traj/episode_001.pb") == data

    def test_write_bytes_creates_parents(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        store.write_bytes("a/b/c.pb", b"x")
        assert tmp_path.joinpath("run", "a", "b", "c.pb").exists()

    def test_write_yaml_copies_file(self, tmp_path: Path):
        source = tmp_path.joinpath("config.yaml")
        source.write_text("key: value\n", encoding="utf-8")
        store = OutputStore(tmp_path.joinpath("run"))
        dest = store.write_yaml("experiment.yaml", source)
        assert dest.read_text(encoding="utf-8") == "key: value\n"

    def test_write_yaml_does_not_modify_source(self, tmp_path: Path):
        source = tmp_path.joinpath("config.yaml")
        source.write_text("original: true\n", encoding="utf-8")
        store = OutputStore(tmp_path.joinpath("run"))
        store.write_yaml("experiment.yaml", source)
        assert source.read_text(encoding="utf-8") == "original: true\n"

    def test_list_empty_prefix_returns_all(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        store.write_json("a.json", {})
        store.write_bytes("b.pb", b"x")
        paths = store.list()
        assert "a.json" in paths
        assert "b.pb" in paths

    def test_list_dirs_empty_when_prefix_missing(self, tmp_path: Path):
        store = OutputStore(tmp_path.joinpath("run"))
        assert store.list_dirs("checkpoints") == []
