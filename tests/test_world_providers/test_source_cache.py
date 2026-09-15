import hashlib
from unittest.mock import Mock, patch

import pytest

from theseo_anysearch.world_providers.api import ProviderInfo, ProviderParameter
from theseo_anysearch.world_providers.source_cache import cached_sources


def test_selectable_resolution_preserves_fixed_provider():
    assert ProviderInfo("fixed", "1", "fixed", 0.25).output_resolution({}) == 0.25
    info = ProviderInfo("variable", "1", "variable", 0.25,
                        (ProviderParameter("meters-per-voxel", "number", default=0.25),),
                        resolution_parameter="meters-per-voxel")
    assert info.output_resolution({"meters-per-voxel": 0.1}) == 0.1
    with pytest.raises(ValueError):
        info.output_resolution({"meters-per-voxel": float("nan")})


def test_verified_download_and_offline_reuse(tmp_path):
    data = b"<robot/>"
    hashes = {"assets/box.urdf": hashlib.sha256(data).hexdigest()}
    response = Mock()
    response.read.return_value = data
    response.geturl.return_value = "https://example.org/assets/box.urdf"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch("theseo_anysearch.world_providers.source_cache.urlopen", return_value=response) as download:
        root = cached_sources(tmp_path, base_url="https://example.org", hashes=hashes)
        assert (root / "assets/box.urdf").read_bytes() == data
        assert cached_sources(tmp_path, base_url="https://example.org", hashes=hashes, offline=True) == root
        assert download.call_count == 1
    (root / "assets/box.urdf").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        cached_sources(tmp_path, base_url="https://example.org", hashes=hashes)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "assets\\escape"])
def test_manifest_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        cached_sources(tmp_path, base_url="https://example.org", hashes={name: "0" * 64})


def test_missing_offline_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        cached_sources(tmp_path, base_url="https://example.org", hashes={"box": "0" * 64}, offline=True)


def test_invalid_download_never_published(tmp_path):
    cache = tmp_path / "cache"
    response = Mock()
    response.read.return_value = b"wrong"
    response.geturl.return_value = "https://example.org/box"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch("theseo_anysearch.world_providers.source_cache.urlopen", return_value=response):
        with pytest.raises(ValueError, match="hash mismatch"):
            cached_sources(cache, base_url="https://example.org", hashes={"box": "0" * 64})
    assert list(cache.iterdir()) == []
