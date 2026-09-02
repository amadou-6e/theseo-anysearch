from pathlib import Path

import pytest

from theseo_anysearch.experiments.custom_geometry import (
    CustomGeometryError,
    GeometryContext,
    GeometryTaskRequirements,
    copy_geometry_source,
    load_geometry_provider,
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
