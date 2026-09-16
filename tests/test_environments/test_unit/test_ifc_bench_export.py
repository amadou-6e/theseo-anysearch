"""Offline tests for the IFC-Bench MEP adapter, built on synthetic IFC fixtures.

No network access and no West Riverside Hospital source files are required:
each fixture is a small, valid IFC4 file constructed at test time.
"""

from __future__ import annotations

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

from theseo_anysearch.environments.ifc_bench_export import (
    DISCIPLINE_FILES,
    build_connectivity,
    confirm_z_up,
    export_discipline,
    extract_elements,
    open_verified_ifc,
    select_task_endpoints,
    verify_upstream,
    _check_step_header,
    _length_scale_to_m,
    _world_aabb_m,
)
from theseo_anysearch.environments.routing_manifests import (
    RoutingTaskRecord,
    RoutingWorldRecord,
    read_sidecar,
)

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


def _build_ifc(
    tmp_path,
    *,
    kind: str,
    with_chain: bool = True,
    with_wall: bool = False,
    correct_elevation: bool = True,
    unit_mm: bool = False,
):
    """Build a small valid IFC4 file with an optional connected pipe-like chain.

    All positions/lengths/radii passed to this function's own inner ``make``
    helper are real meters; when ``unit_mm`` is set, the file's own declared
    unit is millimeters (as real IFC-Bench BIM exports commonly are) and
    those meter values are converted to raw millimeters before being written,
    so the adapter is exercised against a non-trivial length_unit_scale — not
    just the scale=1.0 case a plain-meters fixture can never distinguish from
    a missing scale multiplication.
    """

    raw_per_m = 1000.0 if unit_mm else 1.0

    def raw(value):
        if isinstance(value, tuple):
            return tuple(component * raw_per_m for component in value)
        return value * raw_per_m

    ifc_file = project_api.create_file(version="IFC4")
    project = root_api.create_entity(ifc_file, ifc_class="IfcProject", name="Fixture")
    unit_api.assign_unit(
        ifc_file, length={"is_metric": True, "raw": "MILLIMETERS" if unit_mm else "METERS"}
    )
    body_context = context_api.add_context(ifc_file, context_type="Model")
    body = context_api.add_context(
        ifc_file, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=body_context,
    )

    site = root_api.create_entity(ifc_file, ifc_class="IfcSite", name="Site")
    building = root_api.create_entity(ifc_file, ifc_class="IfcBuilding", name="Building")
    ground = root_api.create_entity(ifc_file, ifc_class="IfcBuildingStorey", name="Ground")
    # edit_object_placement's matrix (is_si=True, the default) and
    # add_profile_representation's depth already take real meters and
    # convert to the file's raw unit internally; only a raw attribute set
    # via create_entity/direct assignment (Elevation, Radius, XDim/YDim)
    # needs this fixture's own raw() conversion.
    ground.Elevation = raw(0.0)
    geometry_api.edit_object_placement(
        ifc_file, product=ground, matrix=_direction_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    upper = root_api.create_entity(ifc_file, ifc_class="IfcBuildingStorey", name="Upper")
    upper.Elevation = raw(3.0)
    upper_elevation_m = 3.0 if correct_elevation else 1.0
    geometry_api.edit_object_placement(
        ifc_file, product=upper,
        matrix=_direction_matrix((0.0, 0.0, upper_elevation_m), (0.0, 0.0, 1.0)),
    )
    aggregate_api.assign_object(ifc_file, products=[site], relating_object=project)
    aggregate_api.assign_object(ifc_file, products=[building], relating_object=site)
    aggregate_api.assign_object(ifc_file, products=[ground, upper], relating_object=building)

    segment_class, fitting_class = {
        "plumbing": ("IfcPipeSegment", "IfcPipeFitting"),
        "electrical": ("IfcCableCarrierSegment", "IfcCableCarrierFitting"),
        "fallback": ("IfcFlowSegment", "IfcFlowFitting"),
    }[kind]

    def make(name, ifc_class, radius, start, direction, length, storey):
        element = root_api.create_entity(ifc_file, ifc_class=ifc_class, name=name)
        spatial_api.assign_container(ifc_file, products=[element], relating_structure=storey)
        profile = ifc_file.create_entity(
            "IfcCircleProfileDef", ProfileType="AREA", Radius=raw(radius)
        )
        geometry_api.add_profile_representation(
            ifc_file, context=body, profile=profile, depth=length
        )
        representations = [
            rep for rep in ifc_file.by_type("IfcShapeRepresentation") if rep.ContextOfItems == body
        ]
        shape = ifc_file.create_entity(
            "IfcProductDefinitionShape", Representations=[representations[-1]]
        )
        element.Representation = shape
        geometry_api.edit_object_placement(
            ifc_file, product=element, matrix=_direction_matrix(start, direction)
        )
        return element

    if with_chain:
        seg_a = make("segA", segment_class, 0.05, (0.0, 2.0, 1.0), (1.0, 0.0, 0.0), 3.0, ground)
        fitting = make("elbow", fitting_class, 0.05, (3.0, 2.0, 1.0), (0.0, 0.0, 1.0), 0.1, ground)
        seg_b = make("segB", segment_class, 0.05, (3.0, 2.0, 1.1), (0.0, 0.0, 1.0), 1.4, upper)
        port_a = system_api.add_port(ifc_file, element=seg_a)
        port_fitting_in = system_api.add_port(ifc_file, element=fitting)
        port_fitting_out = system_api.add_port(ifc_file, element=fitting)
        port_b = system_api.add_port(ifc_file, element=seg_b)
        system_api.connect_port(ifc_file, port_a, port_fitting_in)
        system_api.connect_port(ifc_file, port_fitting_out, port_b)

    if with_wall:
        wall = root_api.create_entity(ifc_file, ifc_class="IfcWall", name="wall")
        spatial_api.assign_container(ifc_file, products=[wall], relating_structure=ground)
        profile = ifc_file.create_entity(
            "IfcRectangleProfileDef", ProfileType="AREA", XDim=raw(4.0), YDim=raw(0.2)
        )
        geometry_api.add_profile_representation(
            ifc_file, context=body, profile=profile, depth=2.5
        )
        representations = [
            rep for rep in ifc_file.by_type("IfcShapeRepresentation") if rep.ContextOfItems == body
        ]
        shape = ifc_file.create_entity(
            "IfcProductDefinitionShape", Representations=[representations[-1]]
        )
        wall.Representation = shape
        geometry_api.edit_object_placement(
            ifc_file, product=wall, matrix=_direction_matrix((10.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )

    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "license.txt").write_text(LICENSE_TEXT, encoding="ascii")
    target_discipline = "electrical" if kind == "electrical" else "plumbing"
    ifc_file.write(str(source_root / DISCIPLINE_FILES[target_discipline]))
    return source_root


def test_typed_plumbing_chain_extracts_without_fallback(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing", with_wall=True)
    ifc_path = source_root / DISCIPLINE_FILES["plumbing"]
    _check_step_header(ifc_path)
    ifc_file = open_verified_ifc(ifc_path)
    elements, fallback_used = extract_elements(ifc_file, discipline="plumbing")
    assert fallback_used is False
    names = sorted(element.Name for element in elements)
    assert names == ["elbow", "segA", "segB"]
    assert "wall" not in names


def test_fallback_used_when_typed_entities_absent(tmp_path):
    source_root = _build_ifc(tmp_path, kind="fallback")
    ifc_path = source_root / DISCIPLINE_FILES["plumbing"]
    ifc_file = open_verified_ifc(ifc_path)
    elements, fallback_used = extract_elements(ifc_file, discipline="plumbing")
    assert fallback_used is True
    assert {element.Name for element in elements} == {"segA", "elbow", "segB"}


def test_missing_typed_and_fallback_entities_raises(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing", with_chain=False, with_wall=True)
    ifc_path = source_root / DISCIPLINE_FILES["plumbing"]
    ifc_file = open_verified_ifc(ifc_path)
    with pytest.raises(ValueError, match="neither"):
        extract_elements(ifc_file, discipline="plumbing")


def test_connectivity_links_only_extracted_elements(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    ifc_path = source_root / DISCIPLINE_FILES["plumbing"]
    ifc_file = open_verified_ifc(ifc_path)
    elements, _ = extract_elements(ifc_file, discipline="plumbing")
    graph, boundary_edges = build_connectivity(ifc_file, elements)
    assert boundary_edges == 0
    by_name = {element.Name: element.GlobalId for element in elements}
    assert graph[by_name["segA"]] == {by_name["elbow"]}
    assert graph[by_name["segB"]] == {by_name["elbow"]}
    assert graph[by_name["elbow"]] == {by_name["segA"], by_name["segB"]}


def test_confirm_z_up_accepts_consistent_storey_placement(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing", correct_elevation=True)
    ifc_file = open_verified_ifc(source_root / DISCIPLINE_FILES["plumbing"])
    scale = _length_scale_to_m(ifc_file)
    assert scale == pytest.approx(1.0)
    confirm_z_up(ifc_file, scale=scale)


def test_confirm_z_up_rejects_inconsistent_storey_placement(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing", correct_elevation=False)
    ifc_file = open_verified_ifc(source_root / DISCIPLINE_FILES["plumbing"])
    scale = _length_scale_to_m(ifc_file)
    with pytest.raises(ValueError, match="differs from"):
        confirm_z_up(ifc_file, scale=scale)


def test_world_aabb_matches_known_extrusion(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    ifc_file = open_verified_ifc(source_root / DISCIPLINE_FILES["plumbing"])
    elements, _ = extract_elements(ifc_file, discipline="plumbing")
    by_name = {element.Name: element for element in elements}
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    low, high = _world_aabb_m(by_name["segA"], settings=settings)
    assert low == pytest.approx((0.0, 1.95, 0.95), abs=0.02)
    assert high == pytest.approx((3.0, 2.05, 1.05), abs=0.02)


def test_wrong_schema_declaration_rejected(tmp_path):
    bad = tmp_path / "bad.ifc"
    bad.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('IFC9999'));\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="schema"):
        _check_step_header(bad)


def test_verify_upstream_rejects_missing_license_notice(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    (source_root / "license.txt").write_text("no license here", encoding="ascii")
    with pytest.raises(ValueError, match="license"):
        verify_upstream(source_root, discipline="plumbing", verify_hashes=False)


def test_verify_upstream_enforces_pinned_hash(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    with pytest.raises(ValueError, match="pinned hash"):
        verify_upstream(source_root, discipline="plumbing", verify_hashes=True)


def test_select_task_endpoints_picks_farthest_leaves(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    ifc_file = open_verified_ifc(source_root / DISCIPLINE_FILES["plumbing"])
    elements, _ = extract_elements(ifc_file, discipline="plumbing")
    graph, _ = build_connectivity(ifc_file, elements)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    centers = {}
    for element in elements:
        low, high = _world_aabb_m(element, settings=settings)
        centers[element.GlobalId] = (np.asarray(low) + np.asarray(high)) / 2.0
    pair, rejections = select_task_endpoints(elements, graph, centers)
    assert pair is not None
    by_name = {element.Name: element.GlobalId for element in elements}
    assert set(pair) == {by_name["segA"], by_name["segB"]}
    assert rejections == []


def test_export_discipline_scales_resolution_correctly_in_millimeter_files(tmp_path):
    """Regression test for a real-data bug: raw profile radii are in the

    file's own declared unit (millimeters for real IFC-Bench exports), not
    meters. Comparing a millimeter fixture against an otherwise-identical
    meter fixture with the same true geometry would silently pass if radius
    scaling were dropped, since a meters-only fixture can never distinguish
    a missing multiplication from a correct one (scale=1.0 either way).
    """

    meters_root = _build_ifc(tmp_path / "m", kind="plumbing", unit_mm=False)
    mm_root = _build_ifc(tmp_path / "mm", kind="plumbing", unit_mm=True)
    meters_report = export_discipline(
        meters_root, tmp_path / "out_m",
        discipline="plumbing", partition="test", verify_hashes=False,
    )
    mm_report = export_discipline(
        mm_root, tmp_path / "out_mm",
        discipline="plumbing", partition="test", verify_hashes=False,
    )
    assert mm_report["meters_per_voxel"] == pytest.approx(meters_report["meters_per_voxel"])
    assert mm_report["body_radius_m"] == pytest.approx(meters_report["body_radius_m"])
    assert mm_report["occupied_voxels"] == meters_report["occupied_voxels"]
    assert len(mm_report["task_ids"]) == 1


def test_export_discipline_produces_valid_bundle(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing", with_wall=True)
    output = tmp_path / "out"
    report = export_discipline(
        source_root, output,
        discipline="plumbing", partition="test", verify_hashes=False,
    )
    assert report["fallback_used"] is False
    assert report["non_mep_clearance_checked"] is False
    assert report["cross_family_holdout_supported"] is False
    assert len(report["task_ids"]) == 1
    assert report["rejected_task_strata"] == []
    assert (output / "source.json").exists()
    assert (output / "world.json").exists()
    assert (output / "task-00.json").exists()
    assert (output / "reference-00.json").exists()
    world_occupancy = np.load(output / "occupancy.npy")
    # Storage axis 1 is vertical; the chain climbs 1.4 m in source Z at its end.
    assert world_occupancy.shape[1] >= 20


def test_grid_frame_round_trips_task_endpoints_to_source_z_up(tmp_path):
    """The stored GridFrame must map storage back to the real, Z-up source point.

    This is the exact defect class that shipped for Gazebo (#477): a default
    identity storage_axes_in_source silently swapping the vertical axis. Here
    the task's own start/goal are round-tripped through the world's frame and
    checked against the known Z-up fixture geometry (segA horizontal in X at
    z=1 m, segB's far tip climbing to z=2.5 m), not merely checked to not crash.
    """

    source_root = _build_ifc(tmp_path, kind="plumbing")
    output = tmp_path / "out"
    export_discipline(source_root, output, discipline="plumbing", partition="test", verify_hashes=False)
    world = read_sidecar(output / "world.json", RoutingWorldRecord)
    task = read_sidecar(output / "task-00.json", RoutingTaskRecord)
    start_source_m = world.frame.to_source_center(task.start_storage, world.extent)
    goal_source_m = world.frame.to_source_center(task.goal_storage, world.extent)
    z_values = sorted((start_source_m[2], goal_source_m[2]))
    # One endpoint sits just below segA at source z~1 m, the other just above
    # segB's climb to z~2.5 m; neither is anywhere near the fixture's x/y span.
    assert z_values[0] == pytest.approx(1.0, abs=0.2)
    assert z_values[1] == pytest.approx(2.5, abs=0.2)


def test_export_discipline_refuses_existing_output(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    output = tmp_path / "out"
    export_discipline(source_root, output, discipline="plumbing", partition="test", verify_hashes=False)
    with pytest.raises(FileExistsError):
        export_discipline(source_root, output, discipline="plumbing", partition="test", verify_hashes=False)


def test_export_discipline_rejects_unreviewed_training_without_evidence(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    output = tmp_path / "out"
    # Training is allowed under CC BY 3.0 evidence; calibration/eval too, so this
    # only exercises the discipline/partition validation, not a rights failure.
    report = export_discipline(
        source_root, output, discipline="plumbing", partition="train", verify_hashes=False,
    )
    assert report["partition"] == "train"


def test_export_discipline_rejects_unknown_discipline(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    with pytest.raises(ValueError, match="discipline"):
        export_discipline(
            source_root, tmp_path / "out",
            discipline="hvac", partition="test", verify_hashes=False,
        )


def test_export_discipline_electrical_uses_cable_carrier_types(tmp_path):
    source_root = _build_ifc(tmp_path, kind="electrical")
    output = tmp_path / "out"
    report = export_discipline(
        source_root, output, discipline="electrical", partition="test", verify_hashes=False,
    )
    assert report["discipline"] == "electrical"
    assert report["fallback_used"] is False
    assert len(report["task_ids"]) == 1
    task = read_sidecar(output / "task-00.json", RoutingTaskRecord)
    assert task.family == "coupled_pipes"


def test_storey_names_restricts_extraction_to_the_named_storey(tmp_path):
    """Ground holds segA/elbow; Upper holds segB only (see _build_ifc)."""

    source_root = _build_ifc(tmp_path, kind="plumbing")
    output = tmp_path / "out"
    report = export_discipline(
        source_root, output, discipline="plumbing", partition="test",
        verify_hashes=False, storey_names=("Ground",),
    )
    assert report["extracted_element_count"] == 2
    assert report["storey_names"] == ["Ground"]
    # elbow's connection to segB (on Upper) is dropped, not silently ignored;
    # connect_port wires both port-pair directions, so one logical connection
    # is two IfcRelConnectsPorts relationships.
    assert report["boundary_connections_excluded"] == 2
    # segA and elbow are still connected to each other within Ground alone,
    # so a task is still produced from just this storey's network.
    assert report["rejected_task_strata"] == []
    assert len(report["task_ids"]) == 1


def test_storey_names_rejects_a_storey_with_no_extracted_elements(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    with pytest.raises(ValueError, match="no extracted elements"):
        export_discipline(
            source_root, tmp_path / "out", discipline="plumbing", partition="test",
            verify_hashes=False, storey_names=("Nonexistent Level",),
        )


def test_crop_bounds_m_filters_elements_and_fixes_world_bounds(tmp_path):
    """A box covering only segA's run excludes elbow and segB."""

    source_root = _build_ifc(tmp_path, kind="plumbing")
    output = tmp_path / "out"
    report = export_discipline(
        source_root, output, discipline="plumbing", partition="test", verify_hashes=False,
        crop_bounds_m=((1.0, 1.9, 0.9), (2.0, 2.1, 1.1)),
    )
    assert report["extracted_element_count"] == 1
    assert report["crop_bounds_m"] == ((1.0, 1.9, 0.9), (2.0, 2.1, 1.1))
    world = read_sidecar(output / "world.json", RoutingWorldRecord)
    # source_origin_m is anchored to the requested box's low corner (allowing
    # for margin), not to the identity (0, 0, 0) origin a data-derived bound
    # from segA's own much larger true AABB (low x=0.0) would produce.
    assert world.frame.source_origin_m[0] != pytest.approx(0.0, abs=1e-6)


def test_crop_bounds_m_rejects_an_inverted_box(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    with pytest.raises(ValueError, match="min < max"):
        export_discipline(
            source_root, tmp_path / "out", discipline="plumbing", partition="test",
            verify_hashes=False,
            crop_bounds_m=((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        )


def test_crop_bounds_m_rejects_an_empty_box(tmp_path):
    source_root = _build_ifc(tmp_path, kind="plumbing")
    with pytest.raises(ValueError, match="no extracted elements"):
        export_discipline(
            source_root, tmp_path / "out", discipline="plumbing", partition="test",
            verify_hashes=False,
            crop_bounds_m=((500.0, 500.0, 500.0), (501.0, 501.0, 501.0)),
        )


def test_verify_route_against_solids_passes_a_clear_route():
    from theseo_anysearch.environments.ifc_bench_export import verify_route_against_solids

    aabbs = {"wall": (np.array([5.0, 5.0, 5.0]), np.array([6.0, 6.0, 6.0]))}
    route = ((0, 0, 0), (1, 0, 0), (2, 0, 0))
    verify_route_against_solids(
        route, aabbs=aabbs, origin_m=np.zeros(3),
        meters_per_voxel=0.1, body_radius_m=0.02,
    )


def test_verify_route_against_solids_independently_catches_a_collision_the_voxel_check_missed():
    """This is the second, continuous-space check's whole reason to exist:

    catch a route that a coarse voxel-grid replay could pass but that
    actually clips a real element's AABB once the discrete cells are mapped
    back to their true continuous-space centers.
    """

    from theseo_anysearch.environments.ifc_bench_export import verify_route_against_solids

    # A storage cell of (5, 0, 0) maps (via the [0,2,1] permutation) to source
    # index (5, 0, 0) -> source-space center (0.55, 0.05, 0.05) at mpv=0.1,
    # origin (0,0,0) -- placed squarely inside this AABB.
    aabbs = {"pipe": (np.array([0.5, 0.0, 0.0]), np.array([0.6, 0.1, 0.1]))}
    route = ((4, 0, 0), (5, 0, 0), (6, 0, 0))
    with pytest.raises(ValueError, match="collides with an extracted element"):
        verify_route_against_solids(
            route, aabbs=aabbs, origin_m=np.zeros(3),
            meters_per_voxel=0.1, body_radius_m=0.02,
        )
