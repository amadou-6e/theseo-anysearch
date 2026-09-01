"""Unit tests for immutable perception-encoder pilot contracts."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from theseo_anysearch.garden.pilots.contracts import (
    AcceleratorCaps,
    ArtifactReference,
    BootstrapRecord,
    DecisionRecord,
    FreshDrawIdentity,
    FrozenPreregistration,
    PilotRunManifest,
    PoolIdentity,
    ResolvedPilotConfig,
    ScoreAnchor,
    SeedAssignments,
    SpecsReference,
    VetoThresholds,
)
from theseo_anysearch.garden.pilots.io import (
    ContractIntegrityError,
    FrozenArtifactError,
    contract_sha256,
    read_contract,
    write_contract,
)


SHA = "a" * 64
GIT_SHA = "b" * 40
SPEC_SHA = "f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d"


def _pool(prefix: str, count: int, observations: int, marker: str) -> PoolIdentity:
    return PoolIdentity(
        geometry_ids=tuple(f"{prefix}-{index:03d}" for index in range(count)),
        observations=observations,
        assignment_sha256=marker * 64,
        query_sha256=marker.upper().lower() * 64,
    )


def _preregistration() -> FrozenPreregistration:
    pools = {
        "pilot_train": _pool("train", 96, 24_000, "1"),
        "pilot_dev_early": _pool("early", 24, 6_000, "2"),
        "pilot_dev_arch": _pool("arch", 24, 6_000, "3"),
        "pilot_dev_interaction": _pool("interaction", 24, 6_000, "4"),
        "pilot_confirm": _pool("confirm", 32, 12_000, "5"),
    }
    anchors = {
        "occupied_iou": ScoreAnchor(
            higher_is_better=True,
            floor=0.40,
            ceiling=0.60,
            floor_source="best-control",
            ceiling_source="supervised-reference",
        ),
        "boundary_f1": ScoreAnchor(
            higher_is_better=True,
            floor=0.30,
            ceiling=0.50,
            floor_source="best-control",
            ceiling_source="supervised-reference",
        ),
        "clearance_nmae": ScoreAnchor(
            higher_is_better=False,
            floor=1.0,
            ceiling=0.7,
            floor_source="best-control",
            ceiling_source="supervised-reference",
        ),
        "reachability_auprc": ScoreAnchor(
            higher_is_better=True,
            floor=0.50,
            ceiling=0.75,
            floor_source="best-control",
            ceiling_source="supervised-reference",
        ),
        "geodesic_nmae": ScoreAnchor(
            higher_is_better=False,
            floor=1.0,
            ceiling=0.75,
            floor_source="best-control",
            ceiling_source="supervised-reference",
        ),
    }
    draws = {
        "P4": FreshDrawIdentity(
            seed=104,
            pool="pilot_dev_arch",
            assignment_sha256=pools["pilot_dev_arch"].assignment_sha256,
            query_sha256=pools["pilot_dev_arch"].query_sha256,
        ),
        "P6": FreshDrawIdentity(
            seed=106,
            pool="pilot_dev_interaction",
            assignment_sha256=pools["pilot_dev_interaction"].assignment_sha256,
            query_sha256=pools["pilot_dev_interaction"].query_sha256,
        ),
        "P7": FreshDrawIdentity(
            seed=107,
            pool="pilot_confirm",
            assignment_sha256=pools["pilot_confirm"].assignment_sha256,
            query_sha256=pools["pilot_confirm"].query_sha256,
        ),
    }
    return FrozenPreregistration(
        frozen_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        specs=SpecsReference(
            repository="https://github.com/amadou-6e/specs",
            commit_sha=SPEC_SHA,
            files=("projects/theseo-anysearch/python/perception-encoder-pilots.md",),
        ),
        accelerator_caps=AcceleratorCaps(
            reference_accelerator="test-device",
            per_pilot_hours={pilot: 1.0 for pilot in (
                "P0", "P1", "P2", "P3", "P4", "P4D", "P5", "P6", "P7", "P8"
            )},
            total_comparative_hours=8.0,
        ),
        seeds=SeedAssignments(),
        vetoes=VetoThresholds(),
        score_anchors=anchors,
        pools=pools,
        fresh_draws=draws,
    )


def _decision(**overrides) -> DecisionRecord:
    values = {
        "pilot": "P1",
        "run_id": "p1-test",
        "preregistration_sha256": SHA,
        "dataset_sha256": "b" * 64,
        "query_sha256": "c" * 64,
        "code_sha": GIT_SHA,
        "development_pool": "pilot_dev_early",
        "locked_inputs": {"seed": 0},
        "candidates": ("candidate-a", "candidate-b"),
        "vetoes": {"candidate-a": (), "candidate-b": ("effective_rank",)},
        "pilot_scores": {"candidate-a": 0.4, "candidate-b": 0.1},
        "bootstrap": BootstrapRecord(
            mean_difference=0.3, lower_95=0.2, upper_95=0.4
        ),
        "learning_curve": {"calibrated": False},
        "resource_metrics": {"accelerator_hours": 0.1},
        "validity_flags": ("short_horizon",),
        "plausible_inversion_limits": ("additional seeds could reverse the ranking",),
        "decision": "winner",
        "retained": ("candidate-a",),
        "rejected": ("candidate-b",),
        "rejection_rules": {"candidate-b": ("effective-rank veto",)},
        "reason": "candidate-a passed all gates",
        "next_pilot": "P2",
        "disposition": "retain",
    }
    values.update(overrides)
    return DecisionRecord(**values)


def test_preregistration_is_complete_and_frozen() -> None:
    preregistration = _preregistration()

    assert len(preregistration.pools["pilot_train"].geometry_ids) == 96
    with pytest.raises(ValidationError):
        preregistration.program = "changed"


def test_preregistration_rejects_geometry_leakage() -> None:
    preregistration = _preregistration()
    raw = preregistration.model_dump()
    early = raw["pools"]["pilot_dev_early"]
    early["geometry_ids"] = (
        preregistration.pools["pilot_train"].geometry_ids[0],
        *early["geometry_ids"][1:],
    )

    with pytest.raises(ValidationError, match="disjoint"):
        FrozenPreregistration.model_validate(raw)


def test_preregistration_rejects_missing_or_placeholder_caps() -> None:
    preregistration = _preregistration()
    raw = preregistration.model_dump()
    del raw["accelerator_caps"]["per_pilot_hours"]["P8"]
    with pytest.raises(ValidationError, match="caps mismatch"):
        FrozenPreregistration.model_validate(raw)

    raw = preregistration.model_dump()
    raw["accelerator_caps"]["per_pilot_hours"]["P8"] = 0
    with pytest.raises(ValidationError, match="must be positive"):
        FrozenPreregistration.model_validate(raw)


def test_score_anchor_rejects_ill_conditioned_denominator() -> None:
    with pytest.raises(ValidationError, match="at least 0.10"):
        ScoreAnchor(
            higher_is_better=True,
            floor=0.5,
            ceiling=0.55,
            floor_source="control",
            ceiling_source="supervised",
        )


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_contract_round_trip_is_content_addressed(tmp_path, suffix: str) -> None:
    contract = _preregistration()
    path = tmp_path / f"preregistration{suffix}"

    identity = write_contract(path, contract)
    loaded = read_contract(path, FrozenPreregistration)

    assert identity == contract_sha256(contract)
    assert loaded == contract
    assert write_contract(path, contract) == identity


def test_frozen_contract_refuses_replacement(tmp_path) -> None:
    path = tmp_path / "decision.json"
    write_contract(path, _decision())

    with pytest.raises(FrozenArtifactError, match="refusing to replace"):
        write_contract(path, _decision(reason="a different result"))


def test_contract_detects_payload_tampering(tmp_path) -> None:
    path = tmp_path / "decision.json"
    write_contract(path, _decision())
    raw = json.loads(path.read_text())
    raw["payload"]["reason"] = "tampered"
    path.write_text(json.dumps(raw))

    with pytest.raises(ContractIntegrityError, match="identity_sha256"):
        read_contract(path, DecisionRecord)


def test_decision_requires_rules_for_every_rejection() -> None:
    with pytest.raises(ValidationError, match="every rejected candidate"):
        _decision(rejection_rules={})


def test_decision_requires_inversion_limits() -> None:
    with pytest.raises(ValidationError):
        _decision(plausible_inversion_limits=())


def test_no_viable_direction_cannot_retain_candidate() -> None:
    with pytest.raises(ValidationError, match="cannot retain"):
        _decision(decision="no_viable_direction")


def test_run_manifest_requires_matching_resolved_pilot() -> None:
    config = ResolvedPilotConfig(
        pilot="P2", candidate="candidate-a", seed=0, update_budget=3_000, batch_size=8
    )
    with pytest.raises(ValidationError, match="must match"):
        PilotRunManifest(
            run_id="run-a",
            pilot="P1",
            integration_base_sha=GIT_SHA,
            code_sha=GIT_SHA,
            preregistration_sha256=SHA,
            dataset_sha256="b" * 64,
            query_sha256="c" * 64,
            resolved_config=config,
            artifacts=(
                ArtifactReference(
                    role="metrics",
                    uri="artifacts/metrics.json",
                    sha256="d" * 64,
                    size_bytes=10,
                    media_type="application/json",
                ),
            ),
        )
