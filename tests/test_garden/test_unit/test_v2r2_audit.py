"""R0 evaluation mechanics only: synthetic records are not pilot evidence."""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from theseo_anysearch.garden.pilots.io import payload_sha256
from theseo_anysearch.garden.pilots.v2r2_audit import (
    CounterfactualContext,
    PredictionFold,
    R0AssessmentSettings,
    STRATA,
    assess_r0,
    conditional_entropy_bounds,
    validate_counterfactual_arrays,
)


def settings(**changes):
    values = dict(
        geometries_per_stratum_per_domain=2,
        counterfactuals_per_context=1024,
        minimum_contexts_per_geometry=1,
        minimum_class_queries=1,
        maximum_conditional_entropy_bits=0.8,
        minimum_visible_skill_bits=0.1,
        maximum_forbidden_skill_bits=0.05,
        maximum_transfer_drop_bits=0.1,
        coverage=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=340,
        probability_clip=1e-6,
        entropy_estimator="plugin_with_exact_binomial_envelope",
        weighting="equal_geometry_then_equal_context",
    )
    values.update(changes)
    return R0AssessmentSettings(**values)


def records():
    rows, folds = [], []
    for domain in ("in_domain", "heldout_generator"):
        ids = []
        for stratum in STRATA:
            for label in (0, 1):
                name = f"{domain}:{stratum}:{label}"
                ids.append(name)
                rows.append(CounterfactualContext(
                    context_id=name, geometry_id=name, fold_id=domain,
                    generator_configuration="A" if domain == "in_domain" else "B",
                    bootstrap_stratum="topology:medium",
                    stratum=stratum, visible_input_sha256=payload_sha256(name),
                    labels=(label,) * 1024,
                    geometric_prior=0.5, coordinates_only=0.5,
                    visible_context=0.95 if label else 0.05, forbidden=0.5,
                    unknown_fraction=0.2,
                ))
        folds.append(PredictionFold(
            fold_id=domain, training_geometry_ids=("train-only",),
            evaluation_geometry_ids=tuple(ids), training_generator_configurations=("A",),
            domain=domain, fit_artifact_sha256="a" * 64,
        ))
    return tuple(rows), tuple(folds)


def change(row, **updates):
    return type(row).model_validate({**row.model_dump(), **updates})


def test_settings_have_no_acceptance_defaults():
    for name in R0AssessmentSettings.model_fields:
        values = settings().model_dump()
        del values[name]
        with pytest.raises(ValidationError):
            R0AssessmentSettings(**values)


@pytest.mark.parametrize("changes", [
    {"bootstrap_seed": True}, {"coverage": float("nan")},
    {"maximum_conditional_entropy_bits": 1}, {"probability_clip": 0.01},
    {"minimum_visible_skill_bits": 0}, {"bootstrap_resamples": 1},
])
def test_invalid_settings_fail(changes):
    with pytest.raises(ValidationError):
        settings(**changes)


def test_array_identity_guard():
    observed = np.zeros((3, 3, 3), dtype=bool)
    observed[0, 0, 0] = True
    unknown = np.zeros_like(observed)
    unknown[1] = True
    completed = np.stack([observed, observed])
    completed[1, 1] = True
    validate_counterfactual_arrays(observed, unknown, completed)
    completed[1, 0, 0, 0] = False
    with pytest.raises(ValueError, match="changed visible"):
        validate_counterfactual_arrays(observed, unknown, completed)
    with pytest.raises(ValueError, match="three-dimensional"):
        validate_counterfactual_arrays(observed[:1], unknown[:1], completed[:, :1])
    observed[unknown] = True
    with pytest.raises(ValueError, match="exposes hidden"):
        validate_counterfactual_arrays(observed, unknown, completed)


def test_entropy_does_not_confuse_constant_finite_samples_with_certainty():
    estimate, lower, upper = conditional_entropy_bounds((0,) * 16, alpha=0.05)
    assert estimate == lower == 0
    assert 0.5 < upper < 1
    assert conditional_entropy_bounds((0, 1) * 8, alpha=0.05)[0] == 1
    assert conditional_entropy_bounds((1,) * 16, alpha=0.05) == pytest.approx((estimate, lower, upper))


def test_success_is_reproducible_and_does_not_authorize_p1():
    rows, folds = records()
    result = assess_r0(settings(), rows, folds)
    assert result == assess_r0(settings(), rows, folds)
    assert result["decision"] == "feasible"
    assert not result["authorizes_comparative_run"]
    digest = result.pop("assessment_payload_sha256")
    assert digest == payload_sha256(result)


def test_missing_stratum_defers_instead_of_dropping_its_weight():
    rows, folds = records()
    result = assess_r0(settings(), tuple(row for row in rows if row.stratum != "6+"), folds)
    assert result["decision"] == "defer"
    assert "6+:heldout_generator:insufficient_support" in result["reasons"]


@pytest.mark.parametrize("alteration,reason", [
    ({"visible_context": 0.5}, "visible_skill"),
    ({"labels": (0, 1) * 512}, "entropy"),
])
def test_unusable_signal_defers(alteration, reason):
    rows, folds = records()
    result = assess_r0(settings(), tuple(change(row, **alteration) for row in rows), folds)
    assert result["decision"] == "defer"
    assert any(reason in value for value in result["reasons"])


def test_forbidden_predictability_and_transfer_fail_separately():
    rows, folds = records()
    forbidden = tuple(change(row, forbidden=row.visible_context) for row in rows)
    result = assess_r0(settings(), forbidden, folds)
    assert any("forbidden_predictability" in reason for reason in result["reasons"])
    transferred = tuple(change(row, visible_context=0.5) if row.fold_id == "heldout_generator" else row for row in rows)
    result = assess_r0(settings(), transferred, folds)
    assert any("generator_transfer" in reason for reason in result["reasons"])


def test_sibling_group_and_generator_fold_guards():
    rows, folds = records()
    with pytest.raises(ValueError, match="cross geometry groups"):
        assess_r0(settings(), (change(rows[0], visible_input_sha256=rows[1].visible_input_sha256), *rows[1:]), folds)
    with pytest.raises(ValueError, match="generator domain"):
        assess_r0(settings(), (change(rows[0], generator_configuration="B"), *rows[1:]), folds)
    with pytest.raises(ValueError, match="completion count"):
        assess_r0(settings(), (change(rows[0], labels=(0, 1)), *rows[1:]), folds)
    with pytest.raises(ValueError, match="multiple evaluation folds"):
        assess_r0(settings(), rows, (folds[0], change(folds[1], evaluation_geometry_ids=(rows[0].geometry_id,))))
    with pytest.raises(ValueError, match="overlap"):
        change(folds[0], training_geometry_ids=(rows[0].geometry_id,))


def test_zero_class_support_is_not_feasible():
    rows, folds = records()
    result = assess_r0(settings(), tuple(change(row, labels=(0,) * 1024) for row in rows), folds)
    assert result["decision"] == "defer"
    assert len(result["reasons"]) == 6


def test_duplicating_queries_does_not_inflate_geometry_support():
    rows, folds = records()
    copies = tuple(change(row, context_id=row.context_id + ":copy") for row in rows)
    result = assess_r0(settings(geometries_per_stratum_per_domain=3), rows + copies, folds)
    assert result["decision"] == "defer"
    assert result["strata"]["1-2"]["domains"]["in_domain"]["geometries"] == 2
