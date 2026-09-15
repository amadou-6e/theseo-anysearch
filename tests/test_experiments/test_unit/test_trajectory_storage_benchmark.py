import json

import pytest

from usage.benchmarks.trajectory_storage.benchmark import synthetic
from usage.benchmarks.trajectory_storage.codecs import (
    FORMATS, OFFSET, RECORD, decode_step, encode_step, read, write,
)


@pytest.mark.parametrize("kind", FORMATS)
@pytest.mark.parametrize("agents", [1, 4])
def test_storage_round_trip_and_cumulative_mutations(tmp_path, kind, agents):
    if kind.endswith("zstd"):
        pytest.importorskip("zstandard")
    payload = synthetic(count=65, agents=agents, mutation_every=16)
    payload["episode"]["steps"][0]["future_field"] = {"nested": [None, "text"]}
    payload["episode"]["steps"][0]["mutations"][0]["future_event"] = "preserved"
    write(tmp_path, kind, payload)
    loaded = read(tmp_path, kind)
    assert json.dumps(loaded.export(), sort_keys=True) == json.dumps(payload, sort_keys=True)
    assert loaded.step(64) == payload["episode"]["steps"][64]
    assert loaded.overlay(64)[(70_000, 1024, 256)] is False
    with pytest.raises(IndexError):
        loaded.step(-1)
    with pytest.raises(IndexError):
        loaded.step(65)


def test_protobuf_preserves_missing_vs_default_and_double_precision():
    for step in [{}, {"action": 0, "placed": False, "mutations": []},
                 {"reward": -.12345678901234567, "cursor_x": 4_294_967_295,
                  "voxel_count": 18_446_744_073_709_551_615, "action": -1}]:
        assert decode_step(encode_step(step)) == step


def test_binary_rejects_truncated_index(tmp_path):
    write(tmp_path, "binary", synthetic(count=3))
    index = tmp_path / "offsets.bin"
    index.write_bytes(index.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        read(tmp_path, "binary")


def test_binary_rejects_out_of_bounds_events(tmp_path):
    write(tmp_path, "binary", synthetic(count=3))
    index = tmp_path / "offsets.bin"
    data = index.read_bytes()
    index.write_bytes(data[:-OFFSET.size] + OFFSET.pack(999_999))
    with pytest.raises(ValueError, match="event offsets"):
        read(tmp_path, "binary")


def test_fixed_record_has_explicit_portable_width():
    assert RECORD.size == 44


@pytest.mark.parametrize("kind", FORMATS)
def test_nonfinite_rewards_are_rejected(tmp_path, kind):
    if kind.endswith("zstd"):
        pytest.importorskip("zstandard")
    payload = synthetic(count=1)
    payload["episode"]["steps"][0]["reward"] = float("nan")
    with pytest.raises(ValueError):
        write(tmp_path, kind, payload)
