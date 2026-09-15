import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "providers/aerialgym/src"))
from anysearch_aerialgym import Provider
from theseo_anysearch.world_providers.bundle import load_bundle


@pytest.mark.parametrize("parameters", [{"meters-per-voxel": 0.13}, {"layout": "unsupported"}, {"body-radius-m": float("nan")}])
def test_bad_parameters_do_not_download(tmp_path, parameters):
    with patch("anysearch_aerialgym.cached_sources") as cache:
        with pytest.raises(ValueError):
            Provider().generate(seed=42, output=tmp_path / "world", parameters=parameters)
        cache.assert_not_called()


def test_negative_seed_does_not_download(tmp_path):
    with patch("anysearch_aerialgym.cached_sources") as cache:
        with pytest.raises(ValueError):
            Provider().generate(seed=-1, output=tmp_path / "world", parameters={})
        cache.assert_not_called()


def test_core_extra_and_provider_entrypoint():
    import tomllib
    root = Path(__file__).resolve().parents[2]
    core = tomllib.loads((root / "pyproject.toml").read_text())
    optional = tomllib.loads((root / "providers/aerialgym/pyproject.toml").read_text())
    assert "theseo-anysearch-aerialgym" in core["project"]["optional-dependencies"]["aerialgym"][0]
    assert optional["project"]["entry-points"]["theseo_anysearch.world_providers"]["aerialgym"] == "anysearch_aerialgym:Provider"


def test_multi_resolution_export_with_independent_box_fixture(tmp_path, monkeypatch):
    from anysearch_aerialgym import SOURCE_HASHES
    from theseo_anysearch.environments.aerial_gym_export import CONFIG
    source = tmp_path / "source"
    hashes = {}
    for name in SOURCE_HASHES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "LICENSE" or name in CONFIG:
            data = b"independent unit-test fixture\n"
        else:
            size = "1 1 1"
            for axis, walls in ((0, ("front", "back")), (1, ("left", "right")), (2, ("top", "bottom"))):
                if any(f"/{wall}_wall.urdf" in name for wall in walls):
                    dims = [10, 10, 6]
                    dims[axis] = 0.1
                    size = " ".join(str(v) for v in dims)
            data = f'<robot name="fixture"><link name="box"><collision><geometry><box size="{size}"/></geometry></collision></link></robot>'.encode()
        path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    worlds = []
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    with patch("anysearch_aerialgym.cached_sources", return_value=source), patch("anysearch_aerialgym.SOURCE_HASHES", hashes):
        from theseo_anysearch.world_providers.service import generate_world, load_verified_world
        with patch("theseo_anysearch.world_providers.service.load_provider", return_value=Provider()):
            generated = tmp_path / "core-generated"
            generate_world("aerialgym", seed=42, output=generated, parameters={"meters-per-voxel": 0.1})
            assert len(list((generated / "previews").glob("*.png"))) == 6
            load_verified_world(generated, use="training")
        for resolution in (0.25, 0.1):
            output = tmp_path / str(resolution)
            Provider().generate(seed=42, output=output, parameters={"meters-per-voxel": resolution})
            worlds.append(load_bundle(output, use="training"))
            assert (output / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    assert worlds[0].world.root_geometry_id == worlds[1].world.root_geometry_id
    assert worlds[0].world.identity_sha256 != worlds[1].world.identity_sha256
    assert worlds[0].world.extent.as_tuple() == (40, 40, 24)
    assert worlds[1].world.extent.as_tuple() == (100, 100, 60)
    assert (worlds[0].root / "scene-instances.json").read_bytes() == (worlds[1].root / "scene-instances.json").read_bytes()
