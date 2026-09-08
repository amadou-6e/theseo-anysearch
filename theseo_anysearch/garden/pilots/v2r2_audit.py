"""R0 counterfactual validation and conservative, geometry-grouped assessment.

These are evaluation primitives, not the complete frozen v2r2 execution protocol.
The caller must freeze the generator, pools, fitting recipes and calibration rules
before opening data. An assessment from this module alone cannot authorize P1.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import beta

from theseo_anysearch.garden.pilots.contracts import FrozenModel, NonEmpty, Sha256
from theseo_anysearch.garden.pilots.io import payload_sha256


STRATA = ("1-2", "3-5", "6+")
CONTROLS = ("geometric_prior", "coordinates_only", "visible_context", "forbidden")


class R0AssessmentSettings(FrozenModel):
    """Required numeric assessment settings; no runtime acceptance defaults."""

    geometries_per_stratum_per_domain: int = Field(strict=True, ge=2)
    counterfactuals_per_context: int = Field(strict=True, ge=2)
    minimum_contexts_per_geometry: int = Field(strict=True, ge=1)
    minimum_class_queries: int = Field(strict=True, ge=1)
    maximum_conditional_entropy_bits: float = Field(gt=0, lt=1, allow_inf_nan=False)
    minimum_visible_skill_bits: float = Field(gt=0, allow_inf_nan=False)
    maximum_forbidden_skill_bits: float = Field(ge=0, allow_inf_nan=False)
    maximum_transfer_drop_bits: float = Field(ge=0, allow_inf_nan=False)
    coverage: float = Field(gt=0.5, lt=1, allow_inf_nan=False)
    bootstrap_resamples: int = Field(strict=True, ge=100)
    bootstrap_seed: int = Field(strict=True, ge=0)
    probability_clip: Literal[0.000001]
    entropy_estimator: Literal["plugin_with_exact_binomial_envelope"]
    weighting: Literal["equal_geometry_then_equal_context"]


class PredictionFold(FrozenModel):
    """A fitting manifest shared by all four controls, including preprocessing."""

    fold_id: NonEmpty
    training_geometry_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    evaluation_geometry_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    training_generator_configurations: tuple[NonEmpty, ...] = Field(min_length=1)
    domain: Literal["in_domain", "heldout_generator"]
    fit_artifact_sha256: Sha256

    @model_validator(mode="after")
    def disjoint(self) -> "PredictionFold":
        train, evaluation = self.training_geometry_ids, self.evaluation_geometry_ids
        if len(set(train)) != len(train) or len(set(evaluation)) != len(evaluation):
            raise ValueError("fold geometry identities must be unique")
        if set(train) & set(evaluation):
            raise ValueError("fitting and evaluation geometries overlap")
        return self


class CounterfactualContext(FrozenModel):
    """One fixed visible observation and query, with repeated hidden completions.

    Predictions are one probability per control, never one per hidden completion.
    The data builder must validate the actual voxel arrays with
    ``validate_counterfactual_arrays`` before constructing this compact record.
    """

    context_id: NonEmpty
    geometry_id: NonEmpty
    fold_id: NonEmpty
    generator_configuration: NonEmpty
    bootstrap_stratum: NonEmpty
    stratum: Literal["1-2", "3-5", "6+"]
    visible_input_sha256: Sha256
    labels: tuple[Literal[0, 1], ...] = Field(min_length=2)
    geometric_prior: float = Field(ge=0, le=1, allow_inf_nan=False)
    coordinates_only: float = Field(ge=0, le=1, allow_inf_nan=False)
    visible_context: float = Field(ge=0, le=1, allow_inf_nan=False)
    forbidden: float = Field(ge=0, le=1, allow_inf_nan=False)
    unknown_fraction: float = Field(gt=0, lt=1, allow_inf_nan=False)


def validate_counterfactual_arrays(
    observed_occupancy: np.ndarray,
    unknown: np.ndarray,
    completions: np.ndarray,
) -> None:
    """Reject changed visible voxels, hidden-value disclosure and invalid shapes."""

    observed, mask, completed = map(np.asarray, (observed_occupancy, unknown, completions))
    if observed.ndim != 3 or any(side < 2 for side in observed.shape):
        raise ValueError("R0 requires a genuinely three-dimensional observed volume")
    if mask.shape != observed.shape or completed.ndim != 4:
        raise ValueError("counterfactual shape mismatch")
    if completed.shape[1:] != observed.shape or len(completed) < 2:
        raise ValueError("at least two aligned completions are required")
    if any(array.dtype != np.bool_ for array in (observed, mask, completed)):
        raise ValueError("counterfactual arrays must be boolean")
    if not mask.any() or mask.all():
        raise ValueError("both observed and unknown voxels are required")
    if observed[mask].any():
        raise ValueError("observed input exposes hidden occupancy")
    if np.any(completed[:, ~mask] != observed[~mask]):
        raise ValueError("counterfactuals changed visible voxels")


def _entropy(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return np.where((p == 0) | (p == 1), 0, -q * np.log2(q) - (1 - q) * np.log2(1 - q))


def conditional_entropy_bounds(labels: tuple[int, ...], *, alpha: float) -> tuple[float, float, float]:
    """Plug-in entropy and an exact-binomial probability-interval envelope.

    Constant finite samples do not imply zero population entropy. This interval
    assumes IID completions conditional on the fixed visible context; a generator
    using correlated MCMC draws must not claim this estimator without adjustment.
    """

    if not labels or any(value not in (0, 1) for value in labels) or not 0 < alpha < 1:
        raise ValueError("binary labels and alpha in (0, 1) are required")
    n, successes = len(labels), sum(labels)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    endpoints = _entropy(np.array([lower, upper]))
    maximum = 1.0 if lower <= 0.5 <= upper else float(endpoints.max())
    return float(_entropy(np.array(successes / n))), float(endpoints.min()), maximum


def _loss(labels: tuple[int, ...], probability: float, clip: float) -> float:
    p = float(np.clip(probability, clip, 1 - clip))
    mean = sum(labels) / len(labels)
    return float(-mean * np.log2(p) - (1 - mean) * np.log2(1 - p))


def assess_r0(
    settings: R0AssessmentSettings,
    contexts: tuple[CounterfactualContext, ...],
    folds: tuple[PredictionFold, ...],
) -> dict[str, object]:
    """Assess held-out predictions without fitting or choosing any control.

    Support gates are checked separately in both generator domains. Complete
    geometry IDs are the bootstrap unit; contexts and sibling completions never
    become independent resampling units. The 21 decision intervals split the
    error budget with the conditional-entropy envelopes (Bonferroni).
    """

    by_fold = {fold.fold_id: fold for fold in folds}
    if len(by_fold) != len(folds):
        raise ValueError("duplicate fold identities")
    if len({row.context_id for row in contexts}) != len(contexts):
        raise ValueError("duplicate context identities")
    ownership: dict[str, str] = {}
    configuration: dict[str, str] = {}
    bootstrap_strata: dict[str, str] = {}
    visible_ownership: dict[str, str] = {}
    for fold in folds:
        for geometry in fold.evaluation_geometry_ids:
            if geometry in ownership:
                raise ValueError("a geometry appears in multiple evaluation folds")
            ownership[geometry] = fold.fold_id
    for row in contexts:
        if row.fold_id not in by_fold or ownership.get(row.geometry_id) != row.fold_id:
            raise ValueError("context is not assigned to its declared evaluation fold")
        old = configuration.setdefault(row.geometry_id, row.generator_configuration)
        if old != row.generator_configuration:
            raise ValueError("siblings disagree on generator configuration")
        old_stratum = bootstrap_strata.setdefault(row.geometry_id, row.bootstrap_stratum)
        if old_stratum != row.bootstrap_stratum:
            raise ValueError("siblings disagree on bootstrap stratum")
        old_group = visible_ownership.setdefault(row.visible_input_sha256, row.geometry_id)
        if old_group != row.geometry_id:
            raise ValueError("identical visible observations cross geometry groups")
        fold = by_fold[row.fold_id]
        seen = row.generator_configuration in fold.training_generator_configurations
        if seen != (fold.domain == "in_domain"):
            raise ValueError("generator domain disagrees with the fitting manifest")
        if len(row.labels) != settings.counterfactuals_per_context:
            raise ValueError("completion count differs from the frozen assessment settings")

    # Half the error budget covers finite-completion entropy; half covers the
    # 21 population decision intervals (seven per stratum). Coverage is approximate.
    entropy_alpha = (1 - settings.coverage) / (2 * max(1, len(contexts)))
    interval_alpha = (1 - settings.coverage) / (2 * 21)
    rng = np.random.default_rng(settings.bootstrap_seed)
    summaries: dict[str, object] = {}
    reasons: list[str] = []

    def draws(matrix: np.ndarray, strata: list[str]) -> np.ndarray:
        result = np.empty((settings.bootstrap_resamples, matrix.shape[1]))
        clusters = [np.flatnonzero(np.asarray(strata) == name) for name in sorted(set(strata))]
        for start in range(0, settings.bootstrap_resamples, 256):
            count = min(256, settings.bootstrap_resamples - start)
            indices = np.concatenate([
                cluster[rng.integers(0, len(cluster), size=(count, len(cluster)))]
                for cluster in clusters
            ], axis=1)
            result[start : start + count] = matrix[indices].mean(axis=1)
        return result

    def interval(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.quantile(values, [interval_alpha / 2, 1 - interval_alpha / 2])]

    for stratum in STRATA:
        domains: dict[str, object] = {}
        domain_draws: dict[str, np.ndarray] = {}
        for domain in ("in_domain", "heldout_generator"):
            rows = [row for row in contexts if row.stratum == stratum and by_fold[row.fold_id].domain == domain]
            groups: dict[str, list[CounterfactualContext]] = {}
            for row in rows:
                groups.setdefault(row.geometry_id, []).append(row)
            positives = sum(sum(row.labels) for row in rows)
            total = sum(len(row.labels) for row in rows)
            supported = (
                len(groups) == settings.geometries_per_stratum_per_domain
                and all(len(group) >= settings.minimum_contexts_per_geometry for group in groups.values())
                and min(positives, total - positives) >= settings.minimum_class_queries
            )
            summary: dict[str, object] = {
                "geometries": len(groups), "contexts": len(rows), "queries": total,
                "positive_queries": positives, "negative_queries": total - positives,
                "support_passed": supported,
            }
            domains[domain] = summary
            if not supported:
                reasons.append(f"{stratum}:{domain}:insufficient_support")
                continue
            matrix = []
            for geometry in sorted(groups):
                records = []
                for row in sorted(groups[geometry], key=lambda item: item.context_id):
                    entropy, lower, upper = conditional_entropy_bounds(row.labels, alpha=entropy_alpha)
                    losses = [_loss(row.labels, getattr(row, name), settings.probability_clip) for name in CONTROLS]
                    records.append([entropy, lower, upper, *losses, row.unknown_fraction])
                matrix.append(np.mean(records, axis=0))
            values = np.asarray(matrix)
            sampled = draws(values, [bootstrap_strata[geometry] for geometry in sorted(groups)])
            domain_draws[domain] = sampled
            means = values.mean(axis=0)
            summary.update({
                "conditional_entropy_bits": float(means[0]),
                "conditional_entropy_interval_bits": [interval(sampled[:, 1])[0], interval(sampled[:, 2])[1]],
                "log_loss_bits": {name: float(means[3 + i]) for i, name in enumerate(CONTROLS)},
                "unknown_fraction": float(means[7]),
            })

        checks: dict[str, object] = {}
        if len(domain_draws) == 2:
            source, target = domain_draws["in_domain"], domain_draws["heldout_generator"]
            # Worst-domain entropy/leakage and weaker-domain visible skill must pass.
            entropy_upper = max(interval(source[:, 2])[1], interval(target[:, 2])[1])
            visible_lower = min(interval(source[:, 3] - source[:, 5])[0], interval(target[:, 3] - target[:, 5])[0])
            forbidden_upper = max(interval(source[:, 3] - source[:, 6])[1], interval(target[:, 3] - target[:, 6])[1])
            transfer_upper = interval((source[:, 3] - source[:, 5]) - (target[:, 3] - target[:, 5]))[1]
            checks = {
                "entropy": {"upper": entropy_upper, "limit": settings.maximum_conditional_entropy_bits, "passed": entropy_upper < settings.maximum_conditional_entropy_bits},
                "visible_skill": {"lower": visible_lower, "limit": settings.minimum_visible_skill_bits, "passed": visible_lower > settings.minimum_visible_skill_bits},
                "forbidden_predictability": {"upper": forbidden_upper, "limit": settings.maximum_forbidden_skill_bits, "passed": forbidden_upper < settings.maximum_forbidden_skill_bits},
                "generator_transfer": {"upper": transfer_upper, "limit": settings.maximum_transfer_drop_bits, "passed": transfer_upper < settings.maximum_transfer_drop_bits},
            }
            for name, check in checks.items():
                if not check["passed"]:
                    reasons.append(f"{stratum}:{name}:insufficient_evidence")
        summaries[stratum] = {"domains": domains, "checks": checks}

    result: dict[str, object] = {
        "kind": "r0_assessment_not_execution_report",
        "decision": "defer" if reasons else "feasible",
        "authorizes_comparative_run": False,
        "reasons": reasons,
        "settings": settings.model_dump(mode="json"),
        "input_sha256": payload_sha256({
            "contexts": [row.model_dump(mode="json") for row in contexts],
            "folds": [fold.model_dump(mode="json") for fold in folds],
        }),
        "strata": summaries,
        "limitations": [
            "Requires IID conditional completions and externally verified fitting provenance.",
            "Cluster-bootstrap coverage is approximate, not a certified confidence guarantee.",
            "Forbidden-feature predictability alone does not prove leakage into encoder inputs.",
            "This assessment is not the frozen protocol or a replacement P1 result.",
        ],
    }
    result["assessment_payload_sha256"] = payload_sha256(result)
    return result
