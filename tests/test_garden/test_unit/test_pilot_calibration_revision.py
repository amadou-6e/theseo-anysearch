"""Unit tests for the P0C calibration-revision amendment contracts (F0)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from theseo_anysearch.garden.pilots.contracts import (
    ArtifactReference,
    RevisedScoreAnchor,
    SeedAssignments,
    SpecsReference,
    SupersededVerdict,
    TrivialityCheck,
    V2R1FrozenPreregistration,
    V2R1VetoThresholds,
    VetoThresholds,
)
from theseo_anysearch.garden.pilots.v2r1 import build_v2r1_pool_identities

SPEC_SHA = "f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d"
SPEC_V2R1 = "0c9e3c633799f5d42b7a603e0845cac0bd494cda"
P0C_REPORT_SHA = "a7a149f9235b38f9ff1f1a230ce791367cf528fc6705e0f987674f1a48d4ea43"


def _triviality(*, passes: bool, min_gain: float = 0.05) -> TrivialityCheck:
    embedding, null = (0.60, 0.10) if passes else (0.12, 0.11)
    gain = embedding - null
    return TrivialityCheck(
        null_input="coordinates_only",
        pvi_embedding=embedding,
        pvi_null=null,
        pvi_gain=gain,
        mdl_embedding_bits=1_000.0,
        mdl_null_bits=4_000.0,
        min_pvi_gain=min_gain,
        passes=gain >= min_gain,
    )


def _revised_anchor(**overrides) -> RevisedScoreAnchor:
    values = dict(
        higher_is_better=True,
        floor=0.40,
        ceiling=0.75,
        floor_source="measured:pca:pilot_calibration",
        ceiling_source="bayes_error:knn:pilot_calibration",
        ceiling_method="bayes_error_knn",
        ceiling_non_collapse_verified=True,
        ceiling_effective_rank_fraction=None,
        triviality=_triviality(passes=True),
        status="active",
        deferral_reason=None,
    )
    values.update(overrides)
    return RevisedScoreAnchor(**values)


def _v1_p1_supersede(**overrides) -> SupersededVerdict:
    values = dict(
        superseded_program="voxel-encoder-pilot-v1",
        superseded_run_id="voxel-encoder-pilot-v1-p1-1",
        superseded_pilot="P1",
        superseded_decision="no_viable_direction",
        fired_veto="false_open_rate_max",
        void_reason=(
            "the false_open_rate > 0.05 veto is unpassable by a fixed random "
            "projection (0.148) under the P0C reachability probe; the v1 P1 "
            "no_viable_direction verdict measured a probe artifact"
        ),
        evidence_run_id="voxel-encoder-pilot-v2-p0-calibration-1",
        evidence_report_sha256=P0C_REPORT_SHA,
        replacement_run_id="voxel-encoder-pilot-v2r1-p1-1",
        recorded_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )
    values.update(overrides)
    return SupersededVerdict(**values)


def _v2r1_preregistration(**overrides) -> V2R1FrozenPreregistration:
    _, pools, draws = build_v2r1_pool_identities(seed=290210)
    artifact = ArtifactReference(
        role="calibration_predictions",
        uri="runtime/perception_encoder/v2r1/p0c-evaluations.json",
        sha256="d" * 64,
        size_bytes=123,
        media_type="application/json",
    )
    anchors = {
        "occupied_iou": _revised_anchor(),
        "boundary_f1": _revised_anchor(floor=0.30, ceiling=0.55),
        "clearance_nmae": _revised_anchor(
            higher_is_better=False,
            floor=1.0,
            ceiling=0.5,
            ceiling_method="knn_residual",
        ),
        "reachability_auprc": _revised_anchor(floor=0.45, ceiling=0.80),
        "geodesic_nmae": _revised_anchor(
            higher_is_better=False,
            floor=0.0225,
            ceiling=0.0482,
            ceiling_method="knn_residual",
            status="deferred",
            deferral_reason=(
                "geodesic target variance below the noise floor at pilot radius; "
                "deferred to Stage 2"
            ),
            triviality=_triviality(passes=False),
        ),
    }
    values = dict(
        dataset_id="voxel-encoder-pilot-v2r1-dataset-1",
        preregistration_id="voxel-encoder-pilot-v2r1-preregistration-1",
        calibration_run_id="voxel-encoder-pilot-v2r1-p0c-1",
        data_sensitivity_run_id="voxel-encoder-pilot-v2r1-p0d-1",
        replacement_p1_run_id="voxel-encoder-pilot-v2r1-p1-1",
        superseded_calibration_run_id="voxel-encoder-pilot-v2-p0-calibration-1",
        superseded_specs_sha=SPEC_SHA,
        frozen_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
        specs=SpecsReference(
            repository="https://github.com/amadou-6e/specs",
            commit_sha=SPEC_V2R1,
            files=("projects/theseo-anysearch/python/perception-encoder-pilots.md",),
        ),
        generator_version="procedural-voxel-generator-v2",
        generator_seed=290210,
        calibration_cap_hours=2.0,
        data_sensitivity_cap_hours=2.0,
        p1_cap_hours=4.0,
        seeds=SeedAssignments(),
        vetoes=V2R1VetoThresholds(
            false_open_rate_max=0.13,
            false_open_baseline=0.15,
            false_open_baseline_name="fixed_random_projection",
        ),
        revised_anchors=anchors,
        active_gate_components=(
            "occupied_iou",
            "boundary_f1",
            "clearance_nmae",
            "reachability_auprc",
        ),
        superseded_verdicts=(_v1_p1_supersede(),),
        pools=pools,
        fresh_draws=draws,
        calibration_artifacts=(artifact,),
        protocol_sha256="e" * 64,
    )
    values.update(overrides)
    return V2R1FrozenPreregistration(**values)


# --- TrivialityCheck --------------------------------------------------------


def test_triviality_check_requires_consistent_gain() -> None:
    with pytest.raises(ValidationError, match="pvi_gain must equal"):
        TrivialityCheck(
            null_input="zeros",
            pvi_embedding=0.5,
            pvi_null=0.1,
            pvi_gain=0.9,
            mdl_embedding_bits=1.0,
            mdl_null_bits=2.0,
            min_pvi_gain=0.05,
            passes=True,
        )


def test_triviality_check_requires_consistent_pass_flag() -> None:
    with pytest.raises(ValidationError, match="passes must reflect"):
        TrivialityCheck(
            null_input="zeros",
            pvi_embedding=0.5,
            pvi_null=0.1,
            pvi_gain=0.4,
            mdl_embedding_bits=1.0,
            mdl_null_bits=2.0,
            min_pvi_gain=0.05,
            passes=False,
        )


# --- RevisedScoreAnchor ---------------------------------------------------------


def test_model_free_ceiling_rejects_an_embedding_rank_fraction() -> None:
    with pytest.raises(ValidationError, match="no embedding rank"):
        _revised_anchor(ceiling_effective_rank_fraction=0.4)


def test_model_free_ceiling_must_declare_non_collapse() -> None:
    with pytest.raises(ValidationError, match="collapse-free by construction"):
        _revised_anchor(ceiling_non_collapse_verified=False)


def test_reference_ceiling_non_collapse_flag_must_match_rank_fraction() -> None:
    with pytest.raises(ValidationError, match="effective-rank"):
        _revised_anchor(
            ceiling_method="regularized_reference",
            ceiling_source="reference:regularized:pilot_calibration",
            ceiling_effective_rank_fraction=0.011,
            ceiling_non_collapse_verified=True,
        )


def test_reference_ceiling_accepts_a_non_collapsed_multitask_trunk() -> None:
    anchor = _revised_anchor(
        ceiling_method="multitask_reference",
        ceiling_source="reference:multitask:pilot_calibration",
        ceiling_effective_rank_fraction=0.42,
        ceiling_non_collapse_verified=True,
    )
    assert anchor.status == "active"


def test_active_anchor_requires_denominator_headroom() -> None:
    with pytest.raises(ValidationError, match="exceed floor by at least 0.10"):
        _revised_anchor(floor=0.95, ceiling=0.99)


def test_active_anchor_requires_usable_information_over_the_null_input() -> None:
    with pytest.raises(ValidationError, match="usable information above the null input"):
        _revised_anchor(triviality=_triviality(passes=False))


def test_deferred_anchor_bypasses_the_denominator_gate() -> None:
    deferred = _revised_anchor(
        higher_is_better=False,
        floor=0.0225,
        ceiling=0.0482,
        ceiling_method="knn_residual",
        status="deferred",
        deferral_reason="target variance below the noise floor; deferred to Stage 2",
        triviality=_triviality(passes=False),
    )
    assert deferred.status == "deferred"


def test_deferred_anchor_requires_a_recorded_reason() -> None:
    with pytest.raises(ValidationError, match="deferred anchor requires a recorded reason"):
        _revised_anchor(status="deferred", deferral_reason=None)


def test_active_anchor_cannot_carry_a_deferral_reason() -> None:
    with pytest.raises(ValidationError, match="cannot carry a deferral reason"):
        _revised_anchor(status="active", deferral_reason="stale")


# --- SupersededVerdict --------------------------------------------------------


def test_superseded_verdict_requires_a_distinct_replacement_run() -> None:
    with pytest.raises(ValidationError, match="new replacement run identity"):
        _v1_p1_supersede(replacement_run_id="voxel-encoder-pilot-v1-p1-1")


def test_superseded_verdict_round_trips_through_json() -> None:
    verdict = _v1_p1_supersede()
    restored = SupersededVerdict.model_validate(verdict.model_dump(mode="json"))
    assert restored == verdict


# --- V2R1FrozenPreregistration ----------------------------------------------


def test_v2r1_preregistration_records_the_v1_p1_supersede() -> None:
    amendment = _v2r1_preregistration()
    assert amendment.schema_version == 3
    assert set(amendment.active_gate_components) == {
        "occupied_iou",
        "boundary_f1",
        "clearance_nmae",
        "reachability_auprc",
    }
    assert amendment.revised_anchors["geodesic_nmae"].status == "deferred"
    supersede = amendment.superseded_verdicts[0]
    assert supersede.fired_veto == "false_open_rate_max"
    assert supersede.evidence_report_sha256 == P0C_REPORT_SHA


def test_v2r1_preregistration_requires_the_v1_p1_supersede_record() -> None:
    with pytest.raises(ValidationError, match="superseded v1 P1 verdict"):
        _v2r1_preregistration(
            superseded_verdicts=(_v1_p1_supersede(superseded_pilot="P2"),)
        )


def test_v2r1_active_gate_components_must_match_active_anchors() -> None:
    with pytest.raises(ValidationError, match="active_gate_components must list"):
        _v2r1_preregistration(active_gate_components=("occupied_iou", "boundary_f1"))


def test_v2r1_requires_every_component_to_have_a_revised_anchor() -> None:
    anchors = dict(_v2r1_preregistration().revised_anchors)
    anchors.pop("occupied_iou")
    with pytest.raises(ValidationError, match="all five components require a revised anchor"):
        _v2r1_preregistration(
            revised_anchors=anchors,
            active_gate_components=("boundary_f1", "clearance_nmae", "reachability_auprc"),
        )


def test_v2r1_template_components_cannot_be_deferred() -> None:
    anchors = dict(_v2r1_preregistration().revised_anchors)
    anchors["boundary_f1"] = _revised_anchor(
        floor=0.30,
        ceiling=0.31,
        status="deferred",
        deferral_reason="not allowed",
        triviality=_triviality(passes=False),
    )
    with pytest.raises(ValidationError, match="calibration template and must stay active"):
        _v2r1_preregistration(
            revised_anchors=anchors,
            active_gate_components=("occupied_iou", "clearance_nmae", "reachability_auprc"),
        )


def test_v2r1_preregistration_is_frozen() -> None:
    amendment = _v2r1_preregistration()
    with pytest.raises(ValidationError):
        amendment.program = "changed"


def test_topology_family_membership_is_by_name_prefix() -> None:
    """reachability*, geodesic* satisfy the rule; a local-only set does not.

    The completed v1/v2/v2r1 contracts are not amended with this rule; it is
    applied by the v2r2 contracts in issue #340. Only the helper is exercised
    here.
    """

    from theseo_anysearch.garden.pilots.contracts import (
        _require_active_topology_component,
    )

    for ok in (
        {"occupied_iou", "reachability"},
        {"occupied_iou", "reachability_auprc"},
        {"occupied_iou", "reachability_logloss_gain"},
        {"occupied_iou", "geodesic_nmae"},
    ):
        _require_active_topology_component(ok)  # no raise

    with pytest.raises(ValueError, match="topology-family component"):
        _require_active_topology_component(
            {"occupied_iou", "boundary_f1", "clearance_nmae"}
        )
