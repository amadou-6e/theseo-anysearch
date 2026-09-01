"""Strict, immutable contracts for perception-encoder pilot runs."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
NonEmpty = Annotated[str, Field(min_length=1)]
PilotName = Literal["P0", "P1", "P2", "P3", "P4", "P4D", "P5", "P6", "P7", "P8"]
Decision = Literal["winner", "tie", "no_viable_direction", "blocked"]
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
