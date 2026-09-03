import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from theseo_anysearch.experiments.custom_geometry import (
    CustomGeometryError,
    GeometryContext,
    GeometryProvider,
    GeometryTaskRequirements,
    _preflight_native_provider,
    copy_geometry_source,
    load_geometry_provider,
    preflight_geometry_provider,
)


class EmptyWorld:
    extent = (16, 16, 16)

    def occupied(self, coordinate):
        return False

    def occupied_in_region(self, minimum, maximum_exclusive):
        return ()


def context() -> GeometryContext:
    return GeometryContext(
        seed=7,
        attempt=1,
        extent=(16, 16, 16),
        task=GeometryTaskRequirements(max_steps=32, action_mode="discrete_6"),
        parameters={"wall_x": 8},
        world=EmptyWorld(),
    )


def test_loads_typed_deterministic_proposal(tmp_path: Path) -> None:
    source = tmp_path.joinpath("geometry.py")
    source.write_text(
        "def wall(context):\n"
        "    x = context.parameters['wall_x']\n"
        "    return {'proposal_id': f'wall-{context.seed}', "
        "'sources': [{'type': 'boxes', 'boxes': [(x, 1, 1, x, 4, 4)]}]}\n",
        encoding="utf-8",
    )
    provider = load_geometry_provider(source, "wall")
    assert provider is not None
    assert provider.generate(context()) == provider.generate(context())
    assert provider.generate(context()).sources[0].type == "boxes"


def test_invalid_output_names_provider(tmp_path: Path) -> None:
    source = tmp_path.joinpath("geometry.py")
    source.write_text("def broken(context):\n    return {'sources': []}\n", encoding="utf-8")
    provider = load_geometry_provider(source, "broken")
    assert provider is not None
    with pytest.raises(CustomGeometryError, match="broken.*invalid proposal"):
        provider.generate(context())


def test_archives_conventional_source(tmp_path: Path) -> None:
    experiment = tmp_path.joinpath("experiment.yaml")
    experiment.write_text("env:\n  geometry:\n    provider:\n      name: wall\n", encoding="utf-8")
    source = tmp_path.joinpath("geometry.py")
    source.write_text("def wall(context): pass\n", encoding="utf-8")
    target = copy_geometry_source(experiment, tmp_path.joinpath("run"), "wall")
    assert target is not None
    assert target.read_bytes() == source.read_bytes()


def _native_provider() -> GeometryProvider:
    return GeometryProvider(
        name="wall",
        source_path=Path("wall.dll"),
        source_sha256="a" * 64,
        generate=lambda context: (_ for _ in ()).throw(
            CustomGeometryError("native geometry providers must be invoked by the environment")
        ),
        native_abi=1,
    )


def _fake_probe(effect: str | Exception) -> MagicMock:
    fake_rust_env = MagicMock()
    if isinstance(effect, Exception):
        fake_rust_env.generate_native_geometry_v1.side_effect = effect
    else:
        fake_rust_env.generate_native_geometry_v1.return_value = effect
    probe = MagicMock()
    probe._rust_env = fake_rust_env
    return probe


def _manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    experiment = tmp_path.joinpath("experiment.yaml")
    experiment.write_text("env:\n  geometry:\n    provider:\n      name: wall\n", encoding="utf-8")
    library = tmp_path.joinpath("libext.so")
    library.write_bytes(b"stub")
    manifest_path = tmp_path.joinpath("extension.json")
    manifest_path.write_text(
        json.dumps(
            {
                "abi_version": 2,
                "source_sha256": "0" * 64,
                "binary_sha256": "0" * 64,
                "library": "libext.so",
                "capabilities": ["geometry"],
                "geometries": ["wall"],
                "platform": "test",
                "machine": "test",
            }
        ),
        encoding="utf-8",
    )
    return experiment, manifest_path


def test_native_preflight_calls_the_rust_abi_once_against_an_empty_probe() -> None:
    payload = json.dumps({"proposal_id": "wall-1", "version": "1", "sources": [], "metadata": {}})
    probe = _fake_probe(payload)
    with patch("theseo_anysearch.environments.gymnasium.voxel_env.VoxelEnv", return_value=probe):
        _preflight_native_provider(
            _native_provider(), (8, 8, 8), seed=1, max_steps=10, action_mode="discrete_6", parameters={},
        )
    probe._rust_env.generate_native_geometry_v1.assert_called_once()
    probe.close.assert_called_once()


def test_native_preflight_surfaces_the_abi_determinism_error(tmp_path: Path) -> None:
    # generate_native_geometry_v1 already enforces determinism in Rust
    # (NativeGeometryV1::invoke_deterministic) and surfaces a mismatch as a
    # ValueError; the preflight wrapper just needs to re-raise it as a
    # CustomGeometryError rather than let it propagate raw.
    probe = _fake_probe(ValueError("native geometry 'wall' is nondeterministic for a fixed context"))
    with patch("theseo_anysearch.environments.gymnasium.voxel_env.VoxelEnv", return_value=probe):
        with pytest.raises(CustomGeometryError, match="wall.*nondeterministic"):
            _preflight_native_provider(
                _native_provider(), (8, 8, 8), seed=1, max_steps=10, action_mode="discrete_6", parameters={},
            )
    probe.close.assert_called_once()


def test_native_preflight_closes_the_probe_on_an_unrelated_failure() -> None:
    probe = _fake_probe(RuntimeError("boom"))
    with patch("theseo_anysearch.environments.gymnasium.voxel_env.VoxelEnv", return_value=probe):
        with pytest.raises(RuntimeError, match="boom"):
            _preflight_native_provider(
                _native_provider(), (8, 8, 8), seed=1, max_steps=10, action_mode="discrete_6", parameters={},
            )
    probe.close.assert_called_once()


def test_preflight_routes_native_providers_through_the_abi_check(tmp_path: Path) -> None:
    experiment, manifest_path = _manifest_fixture(tmp_path)
    geometry = SimpleNamespace(
        provider=SimpleNamespace(name="wall", parameters={}), extent=None, grid_size=8,
    )
    env = SimpleNamespace(seed=1, max_steps=10, action=SimpleNamespace(mode="discrete_6"))
    payload = json.dumps({"proposal_id": "wall-1", "version": "1", "sources": [], "metadata": {}})
    probe = _fake_probe(payload)
    with patch(
        "theseo_anysearch.experiments.native_extensions.discover_native_manifest",
        return_value=manifest_path,
    ), patch("theseo_anysearch.environments.gymnasium.voxel_env.VoxelEnv", return_value=probe):
        provider = preflight_geometry_provider(geometry, env, experiment)
    assert provider is not None
    assert provider.native_abi == 1
    probe._rust_env.generate_native_geometry_v1.assert_called_once()


def test_preflight_native_determinism_failure_is_actionable(tmp_path: Path) -> None:
    experiment, manifest_path = _manifest_fixture(tmp_path)
    geometry = SimpleNamespace(
        provider=SimpleNamespace(name="wall", parameters={}), extent=None, grid_size=8,
    )
    env = SimpleNamespace(seed=1, max_steps=10, action=SimpleNamespace(mode="discrete_6"))
    probe = _fake_probe(ValueError("native geometry 'wall' is nondeterministic for a fixed context"))
    with patch(
        "theseo_anysearch.experiments.native_extensions.discover_native_manifest",
        return_value=manifest_path,
    ), patch("theseo_anysearch.environments.gymnasium.voxel_env.VoxelEnv", return_value=probe):
        with pytest.raises(CustomGeometryError, match="wall.*nondeterministic"):
            preflight_geometry_provider(geometry, env, experiment)
