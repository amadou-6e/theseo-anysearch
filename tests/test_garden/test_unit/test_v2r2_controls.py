"""Synthetic development checks, not preregistered R0 evidence."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from theseo_anysearch.garden.pilots.io import payload_sha256
from theseo_anysearch.garden.pilots.v2r2_controls import (
    R0ControlRecipe, example_from_completions, fit_r0_controls, observed_control_features,
)


def example(index, *, config="A"):
    observed = np.zeros((7, 7, 7), dtype=bool)
    observed[0, 1 + index // 5, 1 + index % 5] = True
    unknown = np.zeros_like(observed)
    unknown[3] = True
    completions = np.stack([observed] * 4)
    completions[2:, 3] = True
    return example_from_completions(
        context_id=f"development-{index}", geometry_id=f"development-{index}",
        generator_configuration=config, bootstrap_stratum="topology:low", stratum="1-2",
        observed_occupancy=observed, unknown=unknown, start=(0, 0, 0), goal=(6, 6, 6),
        completions=completions, forbidden_features=(float(index),),
    )


def recipe(**changes):
    values = dict(
        updates=3, learning_rate=0.01, weight_decay=0.01,
        visible_hidden_width=8, forbidden_hidden_width=4,
        seed=340, maximum_wall_seconds=60,
        optimizer="adamw_full_batch", checkpoint="last_update_no_evaluation_selection",
        weighting="equal_geometry_then_equal_context",
        geometric_prior="linear_observed_density_unknown_density_l1_separation",
    )
    values.update(changes)
    return R0ControlRecipe(**values)


def fit(training=None, evaluation=None, **kwargs):
    return fit_r0_controls(
        training or tuple(example(i) for i in range(4)),
        evaluation or tuple(example(i, config="B") for i in range(4, 6)),
        fold_id="development-fold", domain="heldout_generator", recipe=recipe(),
        device="cpu", **kwargs,
    )


def test_actual_connectivity_targets_and_array_snapshot():
    row = example(0)
    assert row.labels == (1, 1, 0, 0)
    assert not row.observed_occupancy.flags.writeable
    assert not row.unknown.flags.writeable


def test_visible_allowlist_ignores_labels_and_forbidden_metadata():
    row = example(0)
    altered = replace(row, labels=(1, 1), forbidden_features=(9999.0,), generator_configuration="secret")
    before, after = observed_control_features(row), observed_control_features(altered)
    for name in ("visible_context", "coordinates_only", "geometric_prior"):
        np.testing.assert_array_equal(before[name], after[name])
    assert not np.array_equal(before["forbidden"], after["forbidden"])
    assert row.visible_input_sha256 == altered.visible_input_sha256


def test_fit_hash_and_cpu_rng_preserved():
    state = torch.random.get_rng_state().clone()
    result = fit()
    assert torch.equal(state, torch.random.get_rng_state())
    assert result.fold.fit_artifact_sha256 == payload_sha256(result.artifact)
    assert result.resources["candidate_training_updates"] == 0
    assert result.resources["control_training_updates"] == 12
    assert not result.artifact["authorizes_comparative_run"]
    assert not result.artifact["forbidden_features_reach_visible_model"]
    assert all(0 <= row.visible_context <= 1 for row in result.contexts)


def test_evaluation_labels_cannot_change_fitting_or_predictions():
    evaluation = tuple(example(i, config="B") for i in range(4, 6))
    original = fit(evaluation=evaluation)
    changed = fit(evaluation=tuple(replace(row, labels=(1, 1, 1, 1)) for row in evaluation))
    assert original.artifact == changed.artifact
    assert [row.visible_context for row in original.contexts] == [row.visible_context for row in changed.contexts]
    assert original.contexts[0].labels != changed.contexts[0].labels


def test_evaluation_features_do_not_change_train_standardization_or_fit():
    original = fit()
    different = fit(evaluation=tuple(example(i, config="B") for i in range(6, 8)))
    assert original.artifact == different.artifact


def test_group_overlap_and_identical_observation_leakage_rejected():
    with pytest.raises(ValueError, match="overlap"):
        fit(evaluation=(replace(example(5, config="B"), geometry_id="development-0"),))
    with pytest.raises(ValueError, match="cross geometry"):
        fit(evaluation=(replace(example(0, config="B"), geometry_id="other", context_id="other"),))


def test_heldout_generator_validation():
    with pytest.raises(ValueError, match="generator configuration"):
        fit(evaluation=(example(5),))


def test_invalid_endpoint_and_masks_rejected():
    with pytest.raises(ValueError, match="integral"):
        replace(example(0), start=(-1, 0, 0))
    with pytest.raises(ValueError, match="observed free"):
        replace(example(0), start=(3, 0, 0))
    with pytest.raises(ValueError, match="distinct"):
        replace(example(0), start=(6, 6, 6))


def test_cuda_requested_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        fit_r0_controls((example(0),), (example(5, config="B"),),
            fold_id="test", domain="heldout_generator", recipe=recipe(), device="cuda")


def test_expired_budget_does_not_emit_completed_fit():
    with pytest.raises(TimeoutError, match="cap exceeded"):
        fit_r0_controls((example(0),), (example(5, config="B"),),
            fold_id="test", domain="heldout_generator", recipe=recipe(maximum_wall_seconds=1e-9), device="cpu")
