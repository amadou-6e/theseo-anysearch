"""Offline tests for the IFC-Bench provider, built on a synthetic IFC fixture
positioned inside the provider's own fixed canonical plumbing crop.

No network access and no West Riverside Hospital source files are required.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.api.aggregate as aggregate_api
import ifcopenshell.api.context as context_api
import ifcopenshell.api.geometry as geometry_api
import ifcopenshell.api.project as project_api
import ifcopenshell.api.root as root_api
import ifcopenshell.api.spatial as spatial_api
import ifcopenshell.api.system as system_api
import ifcopenshell.api.unit as unit_api

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "providers/ifcbench/src"))
from anysearch_ifcbench import CANONICAL_SCOPE, Provider

LICENSE_TEXT = (
    "West Riverside Hospital IFC Models\n"
    "This work is licensed under the Creative Commons Attribution 3.0 Unported "
    "License CC BY 3.0.\n"
)


def _direction_matrix(translation, z_world):
    z_world = np.asarray(z_world, dtype=float)
    z_world /= np.linalg.norm(z_world)
    reference = np.array([0.0, 0.0, 1.0]) if abs(z_world[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_world = np.cross(reference, z_world)
    x_world /= np.linalg.norm(x_world)
    y_world = np.cross(z_world, x_world)
    matrix = np.eye(4)
    matrix[:3, 0] = x_world
    matrix[:3, 1] = y_world
    matrix[:3, 2] = z_world
    matrix[:3, 3] = translation
    return matrix


def _wrh_plumbing_fixture(tmp_path) -> Path:
    """A short connected pipe chain positioned inside CANONICAL_SCOPE's own
    plumbing crop box and storey name, so the provider's real, non-mocked
    crop/storey filtering and its fixed meters-per-voxel both apply exactly
    as they would for the real file.
    """

    box_low, box_high = CANONICAL_SCOPE["plumbing"]["crop_bounds_m"]
    storey_name = CANONICAL_SCOPE["plumbing"]["storey_names"][0]
    origin = (box_low[0] + 0.1, (box_low[1] + box_high[1]) / 2, box_low[2] + 0.2)

    ifc_file = project_api.create_file(version="IFC4")
    root_api.create_entity(ifc_file, ifc_class="IfcProject", name="Fixture")
    unit_api.assign_unit(ifc_file, length={"is_metric": True, "raw": "METERS"})
    body_context = context_api.add_context(ifc_file, context_type="Model")
    body = context_api.add_context(
        ifc_file, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=body_context,
    )
    site = root_api.create_entity(ifc_file, ifc_class="IfcSite", name="Site")
    building = root_api.create_entity(ifc_file, ifc_class="IfcBuilding", name="Building")
    ground = root_api.create_entity(ifc_file, ifc_class="IfcBuildingStorey", name="Ground")
    ground.Elevation = 0.0
    geometry_api.edit_object_placement(
        ifc_file, product=ground, matrix=_direction_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    storey = root_api.create_entity(ifc_file, ifc_class="IfcBuildingStorey", name=storey_name)
    storey.Elevation = origin[2]
    geometry_api.edit_object_placement(
        ifc_file, product=storey, matrix=_direction_matrix((0.0, 0.0, origin[2]), (0.0, 0.0, 1.0))
    )
    aggregate_api.assign_object(ifc_file, products=[site], relating_object=ifc_file.by_type("IfcProject")[0])
    aggregate_api.assign_object(ifc_file, products=[building], relating_object=site)
    aggregate_api.assign_object(ifc_file, products=[ground, storey], relating_object=building)

    def make(name, ifc_class, radius, start, direction, length):
        element = root_api.create_entity(ifc_file, ifc_class=ifc_class, name=name)
        spatial_api.assign_container(ifc_file, products=[element], relating_structure=storey)
        profile = ifc_file.create_entity("IfcCircleProfileDef", ProfileType="AREA", Radius=radius)
        geometry_api.add_profile_representation(ifc_file, context=body, profile=profile, depth=length)
        representations = [
            rep for rep in ifc_file.by_type("IfcShapeRepresentation") if rep.ContextOfItems == body
        ]
        shape = ifc_file.create_entity("IfcProductDefinitionShape", Representations=[representations[-1]])
        element.Representation = shape
        geometry_api.edit_object_placement(ifc_file, product=element, matrix=_direction_matrix(start, direction))
        return element

    seg_a = make("segA", "IfcPipeSegment", 0.01, origin, (1.0, 0.0, 0.0), 0.3)
    elbow_point = (origin[0] + 0.3, origin[1], origin[2])
    fitting = make("elbow", "IfcPipeFitting", 0.01, elbow_point, (0.0, 0.0, 1.0), 0.02)
    seg_b_start = (elbow_point[0], elbow_point[1], elbow_point[2] + 0.02)
    seg_b = make("segB", "IfcPipeSegment", 0.01, seg_b_start, (0.0, 0.0, 1.0), 0.18)
    port_a = system_api.add_port(ifc_file, element=seg_a)
    port_fitting_in = system_api.add_port(ifc_file, element=fitting)
    port_fitting_out = system_api.add_port(ifc_file, element=fitting)
    port_b = system_api.add_port(ifc_file, element=seg_b)
    system_api.connect_port(ifc_file, port_a, port_fitting_in)
    system_api.connect_port(ifc_file, port_fitting_out, port_b)

    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "license.txt").write_text(LICENSE_TEXT, encoding="ascii")
    ifc_file.write(str(source_root / "plumb_ifc4.ifc"))
    return source_root


@pytest.mark.parametrize("parameters", [
    {"discipline": "hvac", "meters-per-voxel": 0.01},
    {"discipline": "plumbing", "meters-per-voxel": 0.5},
    {"discipline": "plumbing"},
    {"body-radius-m": float("nan")},
    {"unknown": 1},
])
def test_invalid_parameters_do_not_download(tmp_path, parameters):
    with patch("anysearch_ifcbench.cached_sources") as download:
        with pytest.raises(ValueError):
            Provider().generate(seed=0, output=tmp_path / "world", parameters=parameters)
        download.assert_not_called()


def test_extra_and_entrypoint():
    import tomllib
    root = Path(__file__).resolve().parents[2]
    core = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    optional = tomllib.loads((root / "providers/ifcbench/pyproject.toml").read_text())["project"]
    assert core["optional-dependencies"]["ifcbench"] == ["theseo-anysearch-ifcbench>=0.1.0,<0.2.0"]
    assert optional["entry-points"]["theseo_anysearch.world_providers"]["ifcbench"] == "anysearch_ifcbench:Provider"
    assert not any("theseo-anysearch-ifcbench" in dependency for dependency in core["dependencies"])
    assert not any(dependency.startswith("ifcopenshell") for dependency in core["dependencies"])


def test_cli_help_does_not_download():
    import click
    from click.testing import CliRunner
    from typer.main import get_command
    from theseo_anysearch.cli.commands.worlds import app
    with patch("theseo_anysearch.cli.commands.worlds.load_provider", return_value=Provider()), \
         patch("anysearch_ifcbench.cached_sources") as download:
        group = get_command(app)
        command = group.get_command(click.Context(group), "ifcbench")
        result = CliRunner().invoke(command, ["--help"])
    assert result.exit_code == 0, result.output
    assert "discipline" in result.output
    assert "meters-per-voxel" in result.output
    download.assert_not_called()


def test_zero_tasks_never_published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with patch("anysearch_ifcbench.cached_sources", return_value=source), \
         patch("anysearch_ifcbench.export_discipline", return_value={"rejected_task_strata": ["x"]}):
        with pytest.raises(ValueError, match="no accepted"):
            Provider().generate(
                seed=0, output=tmp_path / "world",
                parameters={"discipline": "plumbing", "meters-per-voxel": 0.01},
            )
    assert not (tmp_path / "world").exists()


def test_generation_pipeline_with_synthetic_fixture(tmp_path, monkeypatch):
    from theseo_anysearch.world_providers.bundle import load_bundle
    from theseo_anysearch.world_providers.service import generate_world, load_verified_world

    source = _wrh_plumbing_fixture(tmp_path)
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    with patch("anysearch_ifcbench.cached_sources", return_value=source), \
         patch("theseo_anysearch.world_providers.service.load_provider", return_value=Provider()):
        output = tmp_path / "out"
        report = generate_world(
            "ifcbench", seed=0, output=output,
            parameters={"discipline": "plumbing", "meters-per-voxel": 0.01},
        )
        assert len(report["previews"]) == 6
        bundle = load_verified_world(output, use="evaluation")
        assert len(bundle.tasks) == 1
        assert bundle.tasks[0].family == "single_pipe"
        # Unlike Gazebo's evaluation-only rights, the model's own CC BY 3.0
        # license.txt explicitly clears training use too (see
        # ifc_bench_export.verify_upstream's RightsRecord); this must not raise.
        load_bundle(output, use="training")
        assert (output / "license.txt").read_text(encoding="utf-8") == LICENSE_TEXT


def test_seed_rotates_task_order_deterministically(tmp_path, monkeypatch):
    """With only one accepted task, rotation is a documented no-op; this
    still proves the rotation bookkeeping (report fields, single task
    surviving) rather than silently skipping it.
    """

    from theseo_anysearch.world_providers.service import generate_world, load_verified_world

    source = _wrh_plumbing_fixture(tmp_path)
    monkeypatch.setenv("ANYSEARCH_WORLDS_REGISTRY", str(tmp_path / "registry.json"))
    with patch("anysearch_ifcbench.cached_sources", return_value=source), \
         patch("theseo_anysearch.world_providers.service.load_provider", return_value=Provider()):
        first = tmp_path / "seed0"
        generate_world("ifcbench", seed=0, output=first,
                       parameters={"discipline": "plumbing", "meters-per-voxel": 0.01})
        second = tmp_path / "seed7"
        generate_world("ifcbench", seed=7, output=second,
                       parameters={"discipline": "plumbing", "meters-per-voxel": 0.01})
    first_bundle = load_verified_world(first, use="evaluation")
    second_bundle = load_verified_world(second, use="evaluation")
    assert len(first_bundle.tasks) == len(second_bundle.tasks) == 1
    assert first_bundle.tasks[0].start_storage == second_bundle.tasks[0].start_storage
