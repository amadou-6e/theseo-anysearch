"""Strict, immutable contracts for perception-encoder pilot runs."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
NonEmpty = Annotated[str, Field(min_length=1)]
PilotName = Literal["P0", "P1", "P2", "P3", "P4", "P4D", "P5", "P6", "P7", "P8"]
Decision = Literal[
    "winner", "tie", "no_viable_direction", "blocked", "no_topology_identifiable"
]
Disposition = Literal["promote", "retain", "reject"]

REQUIRED_POOLS = {
    "pilot_train": 96,
    "pilot_dev_early": 24,
    "pilot_dev_arch": 24,
    "pilot_dev_interaction": 24,
    "pilot_confirm": 32,
}
REQUIRED_OBSERVATIONS = {
    "pilot_train": 24_000,
    "pilot_dev_early": 6_000,
    "pilot_dev_arch": 6_000,
    "pilot_dev_interaction": 6_000,
    "pilot_confirm": 12_000,
}
V2_REQUIRED_POOLS = {
    **REQUIRED_POOLS,
    "pilot_calibration": 24,
    "pilot_diagnostic": 24,
}
V2_REQUIRED_OBSERVATIONS = {
    **REQUIRED_OBSERVATIONS,
    "pilot_calibration": 6_000,
    "pilot_diagnostic": 6_000,
}
REQUIRED_SCORE_COMPONENTS = {
    "occupied_iou",
    "boundary_f1",
    "clearance_nmae",
    "reachability_auprc",
    "geodesic_nmae",
}
REQUIRED_CAPS = {"P0", "P1", "P2", "P3", "P4", "P4D", "P5", "P6", "P7", "P8"}


class FrozenModel(BaseModel):
    """Base class for strict values that cannot be mutated after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SpecsReference(FrozenModel):
    """Immutable reference to the exact specifications governing a run."""

    repository: Literal["https://github.com/amadou-6e/specs"]
    commit_sha: GitSha
    files: tuple[NonEmpty, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_repository_paths(self) -> "SpecsReference":
        if len(set(self.files)) != len(self.files):
            raise ValueError("specification paths must be unique")
        if any(path.startswith(("/", "\\")) or ".." in path.split("/") for path in self.files):
            raise ValueError("specification paths must be repository-relative")
        return self


class ArtifactReference(FrozenModel):
    """Content-addressed reference to a run artifact kept outside Git."""

    role: NonEmpty
    uri: NonEmpty
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: NonEmpty = "application/octet-stream"


class AcceleratorCaps(FrozenModel):
    """Compute ceilings committed before comparative results are opened."""

    reference_accelerator: NonEmpty
    per_pilot_hours: dict[str, float]
    total_comparative_hours: float = Field(gt=0)

    @model_validator(mode="after")
    def require_all_positive_caps(self) -> "AcceleratorCaps":
        keys = set(self.per_pilot_hours)
        if keys != REQUIRED_CAPS:
            missing = sorted(REQUIRED_CAPS - keys)
            extra = sorted(keys - REQUIRED_CAPS)
            raise ValueError(f"per-pilot caps mismatch; missing={missing}, extra={extra}")
        if any(value <= 0 for value in self.per_pilot_hours.values()):
            raise ValueError("per-pilot accelerator-hour caps must be positive")
        return self


class SeedAssignments(FrozenModel):
    """Seed allocation fixed by the pilot preregistration."""

    direction_finding: tuple[Literal[0], ...] = (0,)
    conditional_confirmation: tuple[Literal[1], ...] = (1,)
    fresh_confirmation: tuple[Literal[2, 3], Literal[2, 3]] = (2, 3)
    wide_context: tuple[Literal[4], ...] = (4,)

    @model_validator(mode="after")
    def require_exact_seed_map(self) -> "SeedAssignments":
        if (
            self.direction_finding != (0,)
            or self.conditional_confirmation != (1,)
            or self.fresh_confirmation != (2, 3)
            or self.wide_context != (4,)
        ):
            raise ValueError("pilot seeds must remain assigned to 0, 1, 2-3, and 4")
        return self


class VetoThresholds(FrozenModel):
    """Numeric comparative vetoes copied from the pinned specification."""

    effective_rank_fraction_min: float = 0.25
    near_dead_dimensions_fraction_max: float = 0.05
    control_selectivity_min: float = 0.05
    embedding_necessity_margin_min: float = 0.05
    false_open_rate_max: float = 0.05
    false_open_regression_max: float = 0.02
    decisive_score_difference_min: float = 0.03
    bootstrap_resamples: int = 10_000

    @model_validator(mode="after")
    def match_pinned_thresholds(self) -> "VetoThresholds":
        expected = (0.25, 0.05, 0.05, 0.05, 0.05, 0.02, 0.03, 10_000)
        actual = (
            self.effective_rank_fraction_min,
            self.near_dead_dimensions_fraction_max,
            self.control_selectivity_min,
            self.embedding_necessity_margin_min,
            self.false_open_rate_max,
            self.false_open_regression_max,
            self.decisive_score_difference_min,
            self.bootstrap_resamples,
        )
        if actual != expected:
            raise ValueError("veto thresholds differ from the pinned pilot specification")
        return self


class ScoreAnchor(FrozenModel):
    """One fixed baseline floor and supervised ceiling used by pilot-score scaling."""

    higher_is_better: bool
    floor: float
    ceiling: float
    floor_source: NonEmpty
    ceiling_source: NonEmpty

    @model_validator(mode="after")
    def require_well_conditioned_anchor(self) -> "ScoreAnchor":
        if self.higher_is_better:
            if self.ceiling - self.floor < 0.10:
                raise ValueError("higher-is-better ceiling must exceed floor by at least 0.10")
        else:
            if self.floor <= 0 or (self.floor - self.ceiling) / self.floor < 0.20:
                raise ValueError("error ceiling must improve on floor by at least 20 percent")
        return self


class PoolIdentity(FrozenModel):
    """Frozen geometry/query identity for one named pilot pool."""

    geometry_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    observations: int = Field(gt=0)
    assignment_sha256: Sha256
    query_sha256: Sha256

    @model_validator(mode="after")
    def require_unique_geometry_ids(self) -> "PoolIdentity":
        if len(set(self.geometry_ids)) != len(self.geometry_ids):
            raise ValueError("geometry IDs must be unique within a pool")
        return self


class FreshDrawIdentity(FrozenModel):
    """Reproducible identity for a pool first opened by a later pilot."""

    seed: int = Field(ge=0)
    pool: Literal["pilot_dev_arch", "pilot_dev_interaction", "pilot_confirm"]
    assignment_sha256: Sha256
    query_sha256: Sha256


class FrozenPreregistration(FrozenModel):
    """Complete, non-placeholder preregistration frozen before P1."""

    schema_version: Literal[1] = 1
    program: Literal["voxel-encoder-pilot-v1"] = "voxel-encoder-pilot-v1"
    frozen_at: datetime
    specs: SpecsReference
    accelerator_caps: AcceleratorCaps
    seeds: SeedAssignments
    vetoes: VetoThresholds
    score_anchors: dict[str, ScoreAnchor]
    pools: dict[str, PoolIdentity]
    fresh_draws: dict[Literal["P4", "P6", "P7"], FreshDrawIdentity]

    @model_validator(mode="after")
    def require_complete_preregistration(self) -> "FrozenPreregistration":
        if set(self.score_anchors) != REQUIRED_SCORE_COMPONENTS:
            raise ValueError("all five pilot-score floor/ceiling anchors are required")
        if set(self.pools) != set(REQUIRED_POOLS):
            raise ValueError("all named pilot pools are required")
        for name, expected_count in REQUIRED_POOLS.items():
            if len(self.pools[name].geometry_ids) != expected_count:
                raise ValueError(f"{name} must contain {expected_count} geometry IDs")
            if self.pools[name].observations != REQUIRED_OBSERVATIONS[name]:
                raise ValueError(
                    f"{name} must contain {REQUIRED_OBSERVATIONS[name]} base observations"
                )
        all_ids = [geometry_id for pool in self.pools.values() for geometry_id in pool.geometry_ids]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("geometry IDs must be disjoint across pilot pools")
        expected_draws = {
            "P4": "pilot_dev_arch",
            "P6": "pilot_dev_interaction",
            "P7": "pilot_confirm",
        }
        if set(self.fresh_draws) != set(expected_draws):
            raise ValueError("fresh draw identities are required for P4, P6, and P7")
        for pilot, pool in expected_draws.items():
            draw = self.fresh_draws[pilot]
            if draw.pool != pool:
                raise ValueError(f"{pilot} must open {pool}")
            if (
                draw.assignment_sha256 != self.pools[pool].assignment_sha256
                or draw.query_sha256 != self.pools[pool].query_sha256
            ):
                raise ValueError(f"{pilot} fresh draw hashes must match {pool}")
        return self


class T3HealthGate(FrozenModel):
    """Frozen outcome of the two-cell T3 mechanism-health replay."""

    run_id: Literal["voxel-encoder-pilot-v2-t3-diagnostic-1"]
    learning_rates: tuple[Literal[0.0001, 0.0003], Literal[0.0001, 0.0003]] = (
        0.0001,
        0.0003,
    )
    seed: Literal[5] = 5
    updates: Literal[2000] = 2000
    implementation_failure: bool
    mechanism_health_failure: bool
    labels_by_learning_rate: dict[str, tuple[str, ...]]
    shared_labels: tuple[str, ...]
    report: ArtifactReference

    @model_validator(mode="after")
    def require_exact_replay(self) -> "T3HealthGate":
        if self.learning_rates != (0.0001, 0.0003):
            raise ValueError("T3 health replay requires both frozen P1 learning rates")
        if set(self.labels_by_learning_rate) != {"0.0001", "0.0003"}:
            raise ValueError("T3 health labels are required for both learning rates")
        return self


class V2FrozenPreregistration(FrozenModel):
    """Calibration-repair contract frozen before replacement P1 is opened."""

    schema_version: Literal[2] = 2
    program: Literal["voxel-encoder-pilot-v2"] = "voxel-encoder-pilot-v2"
    dataset_id: Literal["voxel-encoder-pilot-v2-dataset-1"]
    preregistration_id: Literal["voxel-encoder-pilot-v2-preregistration-1"]
    calibration_run_id: Literal["voxel-encoder-pilot-v2-p0-calibration-1"]
    diagnostic_run_id: Literal["voxel-encoder-pilot-v2-t3-diagnostic-1"]
    replacement_p1_run_id: Literal["voxel-encoder-pilot-v2-p1-1"]
    superseded_program: Literal["voxel-encoder-pilot-v1"] = "voxel-encoder-pilot-v1"
    superseded_specs_sha: GitSha
    frozen_at: datetime
    specs: SpecsReference
    generator_version: NonEmpty
    generator_seed: int = Field(ge=0)
    calibration_cap_hours: float = Field(gt=0)
    diagnostic_cap_hours: float = Field(gt=0)
    p1_cap_hours: float = Field(gt=0)
    seeds: SeedAssignments
    vetoes: VetoThresholds
    score_anchors: dict[str, ScoreAnchor]
    pools: dict[str, PoolIdentity]
    fresh_draws: dict[Literal["P4", "P6", "P7"], FreshDrawIdentity]
    calibration_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    anchor_selection_without_calibration: Literal[True] = True
    calibration_used_for_candidate_ranking: Literal[False] = False
    t3_health: T3HealthGate

    @model_validator(mode="after")
    def require_complete_v2_preregistration(self) -> "V2FrozenPreregistration":
        if set(self.score_anchors) != REQUIRED_SCORE_COMPONENTS:
            raise ValueError("all five measured v2 score anchors are required")
        if set(self.pools) != set(V2_REQUIRED_POOLS):
            raise ValueError("all seven v2 pilot pools are required")
        for name, expected_count in V2_REQUIRED_POOLS.items():
            pool = self.pools[name]
            if len(pool.geometry_ids) != expected_count:
                raise ValueError(f"{name} must contain {expected_count} geometry IDs")
            if pool.observations != V2_REQUIRED_OBSERVATIONS[name]:
                raise ValueError(
                    f"{name} must contain {V2_REQUIRED_OBSERVATIONS[name]} observations"
                )
        all_ids = [geometry_id for pool in self.pools.values() for geometry_id in pool.geometry_ids]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("geometry IDs must be disjoint across v2 pilot pools")
        if any(geometry_id.startswith("pilot-v1-") for geometry_id in all_ids):
            raise ValueError("v2 pools must be disjoint from opened v1 geometry IDs")
        expected_draws = {
            "P4": "pilot_dev_arch",
            "P6": "pilot_dev_interaction",
            "P7": "pilot_confirm",
        }
        if set(self.fresh_draws) != set(expected_draws):
            raise ValueError("fresh draw identities are required for P4, P6, and P7")
        for pilot, pool_name in expected_draws.items():
            draw = self.fresh_draws[pilot]
            pool = self.pools[pool_name]
            if draw.pool != pool_name:
                raise ValueError(f"{pilot} must open {pool_name}")
            if draw.assignment_sha256 != pool.assignment_sha256:
                raise ValueError(f"{pilot} assignment hash must match {pool_name}")
            if draw.query_sha256 != pool.query_sha256:
                raise ValueError(f"{pilot} query hash must match {pool_name}")
        roles = [artifact.role for artifact in self.calibration_artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("v2 calibration artifact roles must be unique")
        if self.t3_health.implementation_failure or self.t3_health.mechanism_health_failure:
            raise ValueError("replacement P1 cannot be preregistered after a failed T3 health gate")
        return self


# ---------------------------------------------------------------------------
# Calibration revision (F0): amendment schema and superseded-verdict record
# ---------------------------------------------------------------------------

V2R1_DATASET_ID = "voxel-encoder-pilot-v2r1-dataset-1"
V2R1_PREREGISTRATION_ID = "voxel-encoder-pilot-v2r1-preregistration-1"
V2R1_CALIBRATION_RUN_ID = "voxel-encoder-pilot-v2r1-p0c-1"
V2R1_DATA_SENSITIVITY_RUN_ID = "voxel-encoder-pilot-v2r1-p0d-1"
V2R1_P1_RUN_ID = "voxel-encoder-pilot-v2r1-p1-1"

CeilingMethod = Literal[
    "bayes_error_knn",
    "bayes_error_mst",
    "bayes_error_direct",
    "knn_residual",
    "multitask_reference",
    "regularized_reference",
]
MODEL_FREE_CEILING_METHODS = frozenset(
    {"bayes_error_knn", "bayes_error_mst", "bayes_error_direct", "knn_residual"}
)
NullInput = Literal["zeros", "coordinates_only"]
AnchorStatus = Literal["active", "deferred"]
CALIBRATION_TEMPLATE_COMPONENTS = ("boundary_f1", "clearance_nmae")
# v2r2 termination rule (frozen): at least one topology-family component must be
# active and pass its gates. If both the reachability and geodesic families are
# deferred or invalid, the study terminates with decision
# ``no_topology_identifiable`` and the direction-finding pilot does not begin.
# Local geometry alone cannot advance the P1-P8 chain. Family membership is by
# name prefix so ``reachability``, ``reachability_auprc``,
# ``reachability_logloss_gain`` and ``geodesic_nmae`` all count.
TOPOLOGY_COMPONENT_FAMILIES = frozenset({"reachability", "geodesic"})


def _topology_family(component: str) -> str:
    return component.split("_", 1)[0]


def _require_active_topology_component(active_gate_components: set[str]) -> None:
    if not any(
        _topology_family(name) in TOPOLOGY_COMPONENT_FAMILIES
        for name in active_gate_components
    ):
        raise ValueError(
            "v2r2 termination rule: at least one topology-family component "
            "(reachability* or geodesic*) must be active; a preregistration with "
            "both topology families deferred cannot be frozen "
            "(decision no_topology_identifiable)"
        )


class TrivialityCheck(FrozenModel):
    """Usable-information gap between the real embedding and a null input.

    F2 populates the measured quantities. Any component that stays in the
    gate must clear ``pvi_gain >= min_pvi_gain``; a task the null input
    already solves carries no encoder-discriminative signal.
    """

    null_input: NullInput
    pvi_embedding: float
    pvi_null: float
    pvi_gain: float
    mdl_embedding_bits: float = Field(gt=0)
    mdl_null_bits: float = Field(gt=0)
    min_pvi_gain: float = Field(gt=0)
    passes: bool

    @model_validator(mode="after")
    def require_consistent_gain(self) -> "TrivialityCheck":
        if abs(self.pvi_gain - (self.pvi_embedding - self.pvi_null)) > 1e-9:
            raise ValueError("pvi_gain must equal pvi_embedding minus pvi_null")
        if self.passes != (self.pvi_gain >= self.min_pvi_gain):
            raise ValueError("passes must reflect pvi_gain against min_pvi_gain")
        return self


class RevisedScoreAnchor(FrozenModel):
    """Denominator anchor for the amended P0C calibration.

    Extends the v1 anchor with ceiling provenance, a non-collapse check for
    reference-derived ceilings, a triviality gate, and an explicit deferral
    state so a component with no usable headroom is recorded out of the gate
    set instead of silently clamping a degenerate denominator.
    """

    higher_is_better: bool
    floor: float
    ceiling: float
    floor_source: NonEmpty
    ceiling_source: NonEmpty
    ceiling_method: CeilingMethod
    ceiling_non_collapse_verified: bool
    ceiling_effective_rank_fraction: float | None = None
    triviality: TrivialityCheck
    status: AnchorStatus
    deferral_reason: str | None = None

    @model_validator(mode="after")
    def require_valid_revised_anchor(self) -> "RevisedScoreAnchor":
        if self.status == "deferred":
            if not self.deferral_reason:
                raise ValueError("a deferred anchor requires a recorded reason")
            return self
        if self.deferral_reason is not None:
            raise ValueError("an active anchor cannot carry a deferral reason")
        if self.ceiling_method in MODEL_FREE_CEILING_METHODS:
            if self.ceiling_effective_rank_fraction is not None:
                raise ValueError("model-free ceilings have no embedding rank to report")
            if not self.ceiling_non_collapse_verified:
                raise ValueError(
                    "model-free ceilings are collapse-free by construction and must "
                    "record ceiling_non_collapse_verified=true"
                )
        else:
            fraction = self.ceiling_effective_rank_fraction
            if fraction is None or not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    "reference ceilings must report ceiling_effective_rank_fraction in [0, 1]"
                )
            if self.ceiling_non_collapse_verified != (fraction > 0.30):
                raise ValueError(
                    "ceiling_non_collapse_verified must reflect an effective-rank "
                    "fraction above 0.30"
                )
        if not self.ceiling_non_collapse_verified:
            raise ValueError("an active anchor requires a collapse-free ceiling")
        if not self.triviality.passes:
            raise ValueError(
                "an active anchor requires usable information above the null input"
            )
        if self.higher_is_better:
            if self.ceiling - self.floor < 0.10:
                raise ValueError(
                    "active higher-is-better ceiling must exceed floor by at least 0.10"
                )
        elif self.floor <= 0 or (self.floor - self.ceiling) / self.floor < 0.20:
            raise ValueError(
                "active error ceiling must improve on floor by at least 20 percent"
            )
        return self


class SupersededVerdict(FrozenModel):
    """Machine-readable retirement of a prior completed pilot decision.

    Records that a verdict is void because a gate it relied on was later
    shown defective, with the evidence run and its report payload hash.
    """

    superseded_program: NonEmpty
    superseded_run_id: NonEmpty
    superseded_pilot: PilotName
    superseded_decision: Decision
    fired_veto: NonEmpty
    void_reason: NonEmpty
    evidence_run_id: NonEmpty
    evidence_report_sha256: Sha256
    replacement_run_id: NonEmpty
    recorded_at: datetime

    @model_validator(mode="after")
    def require_distinct_replacement(self) -> "SupersededVerdict":
        if self.superseded_run_id == self.replacement_run_id:
            raise ValueError("a superseded verdict requires a new replacement run identity")
        return self


class V2R1MetricPlan(FrozenModel):
    """A probe definition frozen before any v2r1 calibration geometry is opened."""

    higher_is_better: bool
    status: AnchorStatus = "active"
    floor_methods: tuple[NonEmpty, ...] = Field(min_length=1)
    ceiling_method: CeilingMethod
    null_input: NullInput
    min_pvi_gain: float = Field(gt=0)
    minimum_absolute_headroom: float | None = Field(default=None, gt=0)
    minimum_relative_error_reduction: float | None = Field(default=None, gt=0, lt=1)
    deferral_reason: str | None = None

    @model_validator(mode="after")
    def require_directional_gate(self) -> "V2R1MetricPlan":
        if self.status == "deferred":
            if not self.deferral_reason:
                raise ValueError("a deferred metric plan requires a reason")
            return self
        if self.deferral_reason is not None:
            raise ValueError("an active metric plan cannot carry a deferral reason")
        if self.higher_is_better and self.minimum_absolute_headroom is None:
            raise ValueError("higher-is-better plans require absolute headroom")
        if not self.higher_is_better and self.minimum_relative_error_reduction is None:
            raise ValueError("error plans require a relative error reduction")
        return self


class V2R1VetoThresholds(FrozenModel):
    """V2r1 vetoes with the reachability threshold measured by E1."""

    effective_rank_fraction_min: Literal[0.25] = 0.25
    near_dead_dimensions_fraction_max: Literal[0.05] = 0.05
    control_selectivity_min: Literal[0.05] = 0.05
    embedding_necessity_margin_min: Literal[0.05] = 0.05
    false_open_rate_max: float = Field(ge=0, le=1)
    false_open_baseline: float = Field(ge=0, le=1)
    false_open_baseline_name: NonEmpty
    false_open_margin: Literal[0.02] = 0.02
    false_open_regression_max: Literal[0.02] = 0.02
    bootstrap_resamples: Literal[10_000] = 10_000

    @model_validator(mode="after")
    def require_empirical_threshold(self) -> "V2R1VetoThresholds":
        expected = max(0.0, self.false_open_baseline - self.false_open_margin)
        if abs(self.false_open_rate_max - expected) > 1e-9:
            raise ValueError("false_open_rate_max must equal best baseline minus 0.02")
        return self


class V2R1ProtocolPreregistration(FrozenModel):
    """Immutable v2r1 methods and identities frozen before revised P0C."""

    schema_version: Literal[3] = 3
    program: Literal["voxel-encoder-pilot-v2r1"] = "voxel-encoder-pilot-v2r1"
    dataset_id: Literal["voxel-encoder-pilot-v2r1-dataset-1"]
    preregistration_id: Literal["voxel-encoder-pilot-v2r1-preregistration-1"]
    calibration_run_id: Literal["voxel-encoder-pilot-v2r1-p0c-1"]
    data_sensitivity_run_id: Literal["voxel-encoder-pilot-v2r1-p0d-1"]
    replacement_p1_run_id: Literal["voxel-encoder-pilot-v2r1-p1-1"]
    frozen_at: datetime
    specs: SpecsReference
    generator_version: NonEmpty
    generator_seed: int = Field(ge=0)
    calibration_cap_hours: float = Field(gt=0)
    data_sensitivity_cap_hours: float = Field(gt=0)
    p1_cap_hours: float = Field(gt=0)
    metric_plans: dict[str, V2R1MetricPlan]
    active_gate_components: tuple[NonEmpty, ...] = Field(min_length=1)
    superseded_verdicts: tuple[SupersededVerdict, ...] = Field(min_length=1)
    pools: dict[str, PoolIdentity]
    fresh_draws: dict[Literal["P4", "P6", "P7"], FreshDrawIdentity]
    query_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    reachability_false_open_margin: Literal[0.02] = 0.02
    p0d_observation_density_multiplier: Literal[4] = 4

    @model_validator(mode="after")
    def require_frozen_protocol(self) -> "V2R1ProtocolPreregistration":
        if set(self.metric_plans) != REQUIRED_SCORE_COMPONENTS:
            raise ValueError("all five components require a frozen metric plan")
        active = {name for name, plan in self.metric_plans.items() if plan.status == "active"}
        if set(self.active_gate_components) != active:
            raise ValueError("active_gate_components must match active metric plans")
        _require_active_topology_component(active)
        for template in CALIBRATION_TEMPLATE_COMPONENTS:
            if self.metric_plans[template].status != "active":
                raise ValueError(f"{template} is a calibration template and must stay active")
        _validate_v2r1_pools(self.pools, self.fresh_draws)
        if len({artifact.role for artifact in self.query_artifacts}) != len(self.query_artifacts):
            raise ValueError("query artifact roles must be unique")
        if not any(
            verdict.superseded_program == "voxel-encoder-pilot-v1"
            and verdict.superseded_pilot == "P1"
            for verdict in self.superseded_verdicts
        ):
            raise ValueError("the protocol must record the superseded v1 P1 verdict")
        return self


def _validate_v2r1_pools(
    pools: dict[str, PoolIdentity],
    fresh_draws: dict[str, FreshDrawIdentity],
) -> None:
    if set(pools) != set(V2_REQUIRED_POOLS):
        raise ValueError("all seven v2r1 pilot pools are required")
    for name, expected_count in V2_REQUIRED_POOLS.items():
        pool = pools[name]
        if len(pool.geometry_ids) != expected_count:
            raise ValueError(f"{name} must contain {expected_count} geometry IDs")
        if pool.observations != V2_REQUIRED_OBSERVATIONS[name]:
            raise ValueError(f"{name} must contain {V2_REQUIRED_OBSERVATIONS[name]} observations")
    all_ids = [geometry_id for pool in pools.values() for geometry_id in pool.geometry_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("geometry IDs must be disjoint across v2r1 pilot pools")
    if any(not geometry_id.startswith("pilot-v2r1-") for geometry_id in all_ids):
        raise ValueError("v2r1 pools must be fresh and use pilot-v2r1 geometry IDs")
    expected_draws = {"P4": "pilot_dev_arch", "P6": "pilot_dev_interaction", "P7": "pilot_confirm"}
    if set(fresh_draws) != set(expected_draws):
        raise ValueError("fresh draw identities are required for P4, P6, and P7")
    for pilot, pool_name in expected_draws.items():
        draw = fresh_draws[pilot]
        pool = pools[pool_name]
        if draw.pool != pool_name or draw.assignment_sha256 != pool.assignment_sha256 or draw.query_sha256 != pool.query_sha256:
            raise ValueError(f"{pilot} fresh draw hashes must match {pool_name}")


class V2R1FrozenPreregistration(FrozenModel):
    """Measured anchors frozen after P0C and before P0D/P1 are opened."""

    schema_version: Literal[3] = 3
    program: Literal["voxel-encoder-pilot-v2r1"] = "voxel-encoder-pilot-v2r1"
    dataset_id: Literal["voxel-encoder-pilot-v2r1-dataset-1"]
    preregistration_id: Literal["voxel-encoder-pilot-v2r1-preregistration-1"]
    calibration_run_id: Literal["voxel-encoder-pilot-v2r1-p0c-1"]
    data_sensitivity_run_id: Literal["voxel-encoder-pilot-v2r1-p0d-1"]
    replacement_p1_run_id: Literal["voxel-encoder-pilot-v2r1-p1-1"]
    amends_program: Literal["voxel-encoder-pilot-v2"] = "voxel-encoder-pilot-v2"
    superseded_calibration_run_id: Literal["voxel-encoder-pilot-v2-p0-calibration-1"]
    superseded_specs_sha: GitSha
    frozen_at: datetime
    specs: SpecsReference
    generator_version: NonEmpty
    generator_seed: int = Field(ge=0)
    calibration_cap_hours: float = Field(gt=0)
    data_sensitivity_cap_hours: float = Field(gt=0)
    p1_cap_hours: float = Field(gt=0)
    seeds: SeedAssignments
    vetoes: V2R1VetoThresholds
    revised_anchors: dict[str, RevisedScoreAnchor]
    active_gate_components: tuple[NonEmpty, ...] = Field(min_length=1)
    superseded_verdicts: tuple[SupersededVerdict, ...] = Field(min_length=1)
    pools: dict[str, PoolIdentity]
    fresh_draws: dict[Literal["P4", "P6", "P7"], FreshDrawIdentity]
    calibration_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    anchor_selection_without_calibration: Literal[True] = True
    calibration_used_for_candidate_ranking: Literal[False] = False
    protocol_sha256: Sha256

    @model_validator(mode="after")
    def require_complete_amendment(self) -> "V2R1FrozenPreregistration":
        if set(self.revised_anchors) != REQUIRED_SCORE_COMPONENTS:
            raise ValueError("all five components require a revised anchor, active or deferred")
        active = {
            name for name, anchor in self.revised_anchors.items() if anchor.status == "active"
        }
        if set(self.active_gate_components) != active:
            raise ValueError(
                "active_gate_components must list exactly the active revised anchors"
            )
        if not active:
            raise ValueError("the amended calibration must keep at least one component in the gate")
        _require_active_topology_component(active)
        for template in CALIBRATION_TEMPLATE_COMPONENTS:
            if self.revised_anchors[template].status != "active":
                raise ValueError(
                    f"{template} is a calibration template and must stay active"
                )
        _validate_v2r1_pools(self.pools, self.fresh_draws)
        roles = [artifact.role for artifact in self.calibration_artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("calibration artifact roles must be unique")
        if not any(
            verdict.superseded_program == "voxel-encoder-pilot-v1"
            and verdict.superseded_pilot == "P1"
            for verdict in self.superseded_verdicts
        ):
            raise ValueError("the amendment must record the superseded v1 P1 verdict")
        return self


class ResolvedPilotConfig(FrozenModel):
    """Fully resolved inputs for one candidate trial."""

    schema_version: Literal[1] = 1
    pilot: PilotName
    candidate: NonEmpty
    seed: int = Field(ge=0)
    update_budget: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    precision: Literal["fp32", "bf16"] = "fp32"
    compilation_enabled: Literal[False] = False
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class PilotRunManifest(FrozenModel):
    """Immutable identity and artifact inventory for one pilot trial."""

    schema_version: Literal[1] = 1
    run_id: NonEmpty
    pilot: PilotName
    integration_base_sha: GitSha
    code_sha: GitSha
    preregistration_sha256: Sha256
    dataset_sha256: Sha256
    query_sha256: Sha256
    resolved_config: ResolvedPilotConfig
    artifacts: tuple[ArtifactReference, ...] = ()

    @model_validator(mode="after")
    def config_matches_manifest(self) -> "PilotRunManifest":
        if self.resolved_config.pilot != self.pilot:
            raise ValueError("resolved config pilot must match the run manifest")
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique within a run manifest")
        return self


class BootstrapRecord(FrozenModel):
    """Bootstrap settings and interval stored with a pilot decision."""

    resamples: Literal[10_000] = 10_000
    paired_geometry_ids: Literal[True] = True
    stratified: Literal[True] = True
    mean_difference: float
    lower_95: float
    upper_95: float


class RuleDeviation(FrozenModel):
    """Explicit transition to a revised preregistration and new run identity."""

    revised_specs: SpecsReference
    superseded_run_id: NonEmpty
    replacement_run_id: NonEmpty
    reason: NonEmpty

    @model_validator(mode="after")
    def require_new_run_identity(self) -> "RuleDeviation":
        if self.superseded_run_id == self.replacement_run_id:
            raise ValueError("a rule deviation requires a new run identity")
        return self


class DecisionRecord(FrozenModel):
    """Machine-readable, auditable outcome emitted after every pilot."""

    schema_version: Literal[1] = 1
    pilot: PilotName
    run_id: NonEmpty
    preregistration_sha256: Sha256
    dataset_sha256: Sha256
    query_sha256: Sha256
    code_sha: GitSha
    development_pool: NonEmpty
    locked_inputs: dict[str, JsonValue]
    candidates: tuple[NonEmpty, ...] = Field(min_length=1)
    vetoes: dict[str, tuple[NonEmpty, ...]]
    pilot_scores: dict[str, float]
    bootstrap: BootstrapRecord | None
    learning_curve: dict[str, JsonValue]
    resource_metrics: dict[str, JsonValue]
    validity_flags: tuple[str, ...]
    plausible_inversion_limits: tuple[NonEmpty, ...] = Field(min_length=1)
    decision: Decision
    retained: tuple[str, ...]
    rejected: tuple[str, ...]
    rejection_rules: dict[str, tuple[NonEmpty, ...]]
    reason: NonEmpty
    next_pilot: PilotName | None
    disposition: Disposition
    deviation: RuleDeviation | None = None

    @model_validator(mode="after")
    def require_auditable_decision(self) -> "DecisionRecord":
        candidates = set(self.candidates)
        retained = set(self.retained)
        rejected = set(self.rejected)
        if len(candidates) != len(self.candidates):
            raise ValueError("candidate names must be unique")
        if not retained <= candidates or not rejected <= candidates:
            raise ValueError("retained and rejected candidates must be declared candidates")
        if retained & rejected:
            raise ValueError("a candidate cannot be both retained and rejected")
        if set(self.rejection_rules) != rejected:
            raise ValueError("every rejected candidate must name its rejection rules")
        if any(not rules for rules in self.rejection_rules.values()):
            raise ValueError("rejection rule lists cannot be empty")
        if self.decision == "winner" and len(retained) != 1:
            raise ValueError("winner decisions retain exactly one candidate")
        if self.decision == "tie" and len(retained) < 2:
            raise ValueError("tie decisions retain at least two candidates")
        if self.decision == "no_viable_direction" and retained:
            raise ValueError("no_viable_direction cannot retain a candidate")
        if self.decision != "blocked" and retained | rejected != candidates:
            raise ValueError("non-blocked decisions must account for every candidate")
        return self


def model_payload(model: BaseModel) -> dict[str, Any]:
    """Return a JSON-compatible payload for canonical serialization."""

    return model.model_dump(mode="json")
