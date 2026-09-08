"""Two-stage v2r2 registration: audit first, comparison only after measured gates."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from theseo_anysearch.garden.pilots.contracts import (
    FrozenModel, GitSha, Sha256, _require_active_topology_component,
)
from theseo_anysearch.garden.pilots.io import payload_sha256
from theseo_anysearch.garden.pilots.v2r2_audit import R0AssessmentSettings
from theseo_anysearch.garden.pilots.v2r2_controls import R0ControlRecipe
from theseo_anysearch.garden.pilots.v2r2_data import identity_plan


class CalibrationRecipe(FrozenModel):
    candidate_probe: Literal["global192_plus_coordinates6_mlp128"]
    raw_control: Literal["conv3d_2_8_16_32_globalmean_coords_mlp144"]
    empirical_reference: Literal["conv3d_2_32_64_128_globalmean_coords_mlp512"]
    input_information: Literal["observed_occupancy_unknown_and_query_coordinates_only"]
    normalization: Literal["binary_voxels_unit_coordinates_no_fitted_normalization"]
    optimizer: Literal["adamw_full_batch"]
    updates: int = Field(strict=True, ge=1)
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    weight_decay: float = Field(ge=0, allow_inf_nan=False)
    selection: Literal["last_update_no_evaluation_selection"]
    menu: tuple[Literal["conditional_log_loss_gain", "occlusion_auprc"], ...]
    metric_rule: Literal["conditional_log_loss_gain_only_no_fallback"]
    minimum_headroom: Literal[0.1]
    probability_clip: Literal[0.000001]
    anchor_count: int = Field(strict=True, ge=1)
    anchor_selection: Literal["hash_rank_observed_free_cells"]
    threshold_grid: tuple[float, ...] = Field(min_length=2)
    threshold_selection: Literal["training_geometry_threefold_max_balanced_accuracy_lowest_tie"]
    minimum_negative_support: int = Field(strict=True, ge=1)
    false_merge_rule: Literal["min_control_reference_upper95_plus_margin_capped1"]
    false_merge_margin: float = Field(gt=0, le=1, allow_inf_nan=False)
    veto_uncertainty: Literal["geometry_cluster_bootstrap_upper95"]
    selectivity_minimum_component_normalized_gain: float = Field(gt=0, allow_inf_nan=False)
    embedding_necessity_minimum_component_normalized_gain: float = Field(gt=0, allow_inf_nan=False)
    retention_rule: Literal["all_four_components_required_geodesic_deferred"]
    p0d_density_multiplier: Literal[4]
    p0d_maximum_headroom_change: float = Field(gt=0, allow_inf_nan=False)
    p0d_maximum_false_merge_change: float = Field(gt=0, allow_inf_nan=False)
    model_free_reference: Literal["diagnostic_only_geometry_disjoint_knn_k15_visible_features"]
    local_templates: Literal["v2r1_masked_occupied_iou_boundary_f1_clearance_nmae"]

    @model_validator(mode="after")
    def complete_menu(self):
        if set(self.menu) != {"conditional_log_loss_gain", "occlusion_auprc"} or len(self.menu) != 2:
            raise ValueError("both metric forms must be declared; no post-hoc fallback")
        if any(not 0 < value < 1 for value in self.threshold_grid) or tuple(sorted(set(self.threshold_grid))) != self.threshold_grid:
            raise ValueError("invalid fixed decision-threshold grid")
        return self


class V2R2AuditProtocol(FrozenModel):
    schema_version: Literal[4]
    program: Literal["voxel-encoder-pilot-v2r2"]
    protocol_id: Literal["voxel-encoder-pilot-v2r2-audit-protocol-1"]
    dataset_id: Literal["voxel-encoder-pilot-v2r2-dataset-1"]
    r0_run_id: Literal["voxel-encoder-pilot-v2r2-r0-1"]
    downstream_ids: tuple[str, ...]
    governing_spec_sha: Literal["0060366f43892bade5acf734cf2e189d8a666ad2"]
    execution_addendum_spec_sha: GitSha
    foundation_sha: Literal["f03509f926d217cf8a0c8c34cd0da38020a5034c"]
    predecessor_spec_sha: Literal["0c9e3c633799f5d42b7a603e0845cac0bd494cda"]
    executable_sha: GitSha
    generator: Literal["native-v2-visible-conditioned-patches-v1"]
    root_seed: int = Field(strict=True, ge=0)
    training_geometries_per_stratum: int = Field(strict=True, ge=12)
    donor_geometries_per_configuration: int = Field(strict=True, ge=16)
    donor_neighbours: int = Field(strict=True, ge=2)
    query_attempts: int = Field(strict=True, ge=1)
    configurations: Literal["A_native_r8_B_native_r16_stride2"]
    conditioning: Literal["visible_hamming_topk_uniform_iid_with_replacement"]
    r0_strata: Literal["observed_optimistic_lexicographic_path_span_no_label_conditioning"]
    r1_positive_strata: Literal["completed_lexicographic_shortest_path_span"]
    r1_negative_strata: Literal["observed_optimistic_path_span_single_bin"]
    stratum_weights: tuple[float, float, float]
    pools: dict[str, list[dict[str, JsonValue]]]
    pool_plan_sha256: Sha256
    query_plan_sha256: Sha256
    assessment: R0AssessmentSettings
    controls: R0ControlRecipe
    calibration: CalibrationRecipe
    r0_wall_cap_seconds: float = Field(gt=0, allow_inf_nan=False)
    r0_accelerator_hour_cap: float = Field(gt=0, allow_inf_nan=False)
    p0c_accelerator_hour_cap: float = Field(gt=0, allow_inf_nan=False)
    p0d_accelerator_hour_cap: float = Field(gt=0, allow_inf_nan=False)
    p1_accelerator_hour_cap: float = Field(gt=0, allow_inf_nan=False)
    workers: int = Field(strict=True, ge=1, le=8)
    required_topology_components: tuple[Literal["reachability"], ...]

    @model_validator(mode="after")
    def validate_registration(self):
        _require_active_topology_component(set(self.required_topology_components))
        if self.required_topology_components != ("reachability",):
            raise ValueError("v2r2 requires reachability; geodesic remains deferred")
        if self.stratum_weights != (1 / 3, 1 / 3, 1 / 3):
            raise ValueError("this execution freezes equal stratum weights")
        for count in (self.training_geometries_per_stratum, self.assessment.geometries_per_stratum_per_domain, self.donor_geometries_per_configuration):
            if count % 12:
                raise ValueError("native pools must balance all twelve family/density strata")
        if self.donor_neighbours > self.donor_geometries_per_configuration:
            raise ValueError("donor neighbour count exceeds bank size")
        expected = identity_plan(self.root_seed, self.training_geometries_per_stratum,
            self.assessment.geometries_per_stratum_per_domain, self.donor_geometries_per_configuration)
        if self.pools != expected or self.pool_plan_sha256 != payload_sha256(expected):
            raise ValueError("pool identity plan changed")
        query_plan = {"pool_plan_sha256": self.pool_plan_sha256, "seed": self.root_seed,
            "r0_rule": self.r0_strata, "attempts": self.query_attempts,
            "r1_positive": self.r1_positive_strata, "r1_negative": self.r1_negative_strata,
            "generator": self.generator}
        if self.query_plan_sha256 != payload_sha256(query_plan):
            raise ValueError("query plan changed")
        expected_ids = tuple(f"voxel-encoder-pilot-v2r2-{name}-1" for name in ("preregistration", "p0c", "p0d", "p1"))
        if self.downstream_ids != expected_ids:
            raise ValueError("fresh downstream run identities required")
        return self


class V2R2ComparativeContract(FrozenModel):
    """Data-dependent comparison fields; instantiate only from verified reports."""

    schema_version: Literal[4]
    preregistration_id: Literal["voxel-encoder-pilot-v2r2-preregistration-1"]
    audit_protocol_sha256: Sha256
    r0_report_payload_sha256: Sha256
    p0c_report_payload_sha256: Sha256
    executable_sha: GitSha
    spec_sha: GitSha
    r0_decision: Literal["feasible"]
    primary_metric: Literal["conditional_log_loss_gain"]
    active_gate_components: tuple[str, ...]
    per_component_floor: dict[str, float]
    per_component_reference: dict[str, float]
    reachability_headroom_bits: float = Field(ge=0.1, allow_inf_nan=False)
    false_merge_veto: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def topology_required(self):
        import math
        _require_active_topology_component(set(self.active_gate_components))
        expected = {"occupied_iou", "boundary_f1", "clearance_nmae", "reachability"}
        if set(self.active_gate_components) != expected or len(self.active_gate_components) != 4:
            raise ValueError("all four active components are required")
        if set(self.per_component_floor) != expected or set(self.per_component_reference) != expected:
            raise ValueError("anchors missing active components")
        if any(not math.isfinite(x) for x in (*self.per_component_floor.values(), *self.per_component_reference.values())):
            raise ValueError("nonfinite anchor")
        gap = self.per_component_floor["reachability"] - self.per_component_reference["reachability"]
        if not math.isclose(gap, self.reachability_headroom_bits, abs_tol=1e-12):
            raise ValueError("measured reachability denominator differs from anchors")
        for component in ("occupied_iou", "boundary_f1"):
            floor, reference = self.per_component_floor[component], self.per_component_reference[component]
            if not 0 <= floor <= reference <= 1 or reference - floor < 0.1:
                raise ValueError("local classification anchor lacks headroom")
        floor, reference = self.per_component_floor["clearance_nmae"], self.per_component_reference["clearance_nmae"]
        if not 0 <= reference <= 0.8 * floor or floor <= 0:
            raise ValueError("clearance anchor lacks headroom")
        return self


def comparative_from_reports(protocol: V2R2AuditProtocol, r0: dict, p0c: dict) -> V2R2ComparativeContract:
    """Verify the evidence chain before constructing data-dependent registration."""
    from theseo_anysearch.garden.pilots.io import contract_sha256
    for report in (r0, p0c):
        actual = payload_sha256({key: value for key, value in report.items() if key != "report_payload_sha256"})
        if actual != report.get("report_payload_sha256"):
            raise ValueError("report payload hash mismatch")
        if report.get("protocol_sha256") != contract_sha256(protocol) or report.get("dataset_id") != protocol.dataset_id:
            raise ValueError("report belongs to another protocol/dataset")
    if r0.get("run_id") != protocol.r0_run_id or r0.get("status") != "completed" or r0.get("r0_decision") != "feasible":
        raise ValueError("R0 has not qualified; comparative freeze prohibited")
    if p0c.get("run_id") != "voxel-encoder-pilot-v2r2-p0c-1" or p0c.get("status") != "passed":
        raise ValueError("P0C has not qualified")
    if p0c.get("r0_report_payload_sha256") != r0["report_payload_sha256"]:
        raise ValueError("P0C does not reference the qualifying R0")
    return V2R2ComparativeContract(
        schema_version=4, preregistration_id="voxel-encoder-pilot-v2r2-preregistration-1",
        audit_protocol_sha256=contract_sha256(protocol), r0_report_payload_sha256=r0["report_payload_sha256"],
        p0c_report_payload_sha256=p0c["report_payload_sha256"], executable_sha=p0c["code_sha"],
        spec_sha=protocol.execution_addendum_spec_sha, r0_decision="feasible", primary_metric="conditional_log_loss_gain",
        active_gate_components=tuple(p0c["active_gate_components"]),
        per_component_floor=p0c["per_component_floor"], per_component_reference=p0c["per_component_reference"],
        reachability_headroom_bits=p0c["reachability_headroom_bits"], false_merge_veto=p0c["false_merge_veto"],
    )
