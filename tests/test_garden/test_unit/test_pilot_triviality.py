"""Unit tests for probe-triviality instrumentation (F2)."""
from __future__ import annotations

import numpy as np
import pytest

from theseo_anysearch.garden.evaluation.triviality import assess_triviality


def _coords(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, 3))


def test_a_task_the_null_input_already_solves_has_no_pvi_gain() -> None:
    # occupied_iou failure mode: the label is a function of the query coordinates,
    # so a coordinates-only null input carries the same information as the
    # embedding.
    coords = _coords(600, seed=0)
    label = (coords[:, 0] > 0.0).astype(int)
    embedding = np.concatenate(
        [coords, np.random.default_rng(1).normal(size=(600, 8))], axis=1
    )
    result = assess_triviality(
        embedding, coords, label, task_type="binary", null_input="coordinates_only"
    )
    assert result.pvi_gain < 0.05
    assert result.passes is False


def test_a_task_only_the_embedding_solves_has_a_clear_pvi_gain() -> None:
    rng = np.random.default_rng(2)
    coords = _coords(600, seed=2)
    hidden = rng.normal(size=(600, 6))
    # label depends on a latent direction present only in the embedding.
    label = (hidden @ rng.normal(size=6) > 0.0).astype(int)
    embedding = np.concatenate([coords, hidden], axis=1)
    result = assess_triviality(
        embedding, coords, label, task_type="binary", null_input="coordinates_only"
    )
    assert result.pvi_gain > 0.1
    assert result.passes is True


def test_zeros_null_input_carries_no_information() -> None:
    rng = np.random.default_rng(3)
    embedding = rng.normal(size=(500, 10))
    label = (embedding @ rng.normal(size=10) > 0).astype(int)
    zeros = np.zeros((500, 1))
    result = assess_triviality(
        embedding, zeros, label, task_type="binary", null_input="zeros"
    )
    assert result.pvi_null < 0.05
    assert result.pvi_gain > 0.1
    assert result.passes is True


def test_regression_triviality_gain_for_a_smooth_latent_target() -> None:
    rng = np.random.default_rng(4)
    coords = _coords(600, seed=4)
    hidden = rng.normal(size=(600, 4))
    target = np.sin(hidden[:, 0]) + 0.05 * rng.normal(size=600)
    embedding = np.concatenate([coords, hidden], axis=1)
    result = assess_triviality(
        embedding,
        coords,
        target,
        task_type="regression",
        null_input="coordinates_only",
    )
    assert result.pvi_gain > 0.0
    assert result.mdl_embedding_bits < result.mdl_null_bits


def test_triviality_result_maps_onto_the_contract_fields() -> None:
    from theseo_anysearch.garden.pilots.contracts import TrivialityCheck

    rng = np.random.default_rng(5)
    embedding = rng.normal(size=(400, 8))
    label = (embedding @ rng.normal(size=8) > 0).astype(int)
    result = assess_triviality(
        embedding,
        np.zeros((400, 1)),
        label,
        task_type="binary",
        null_input="zeros",
    )
    check = TrivialityCheck(
        null_input=result.null_input,
        pvi_embedding=result.pvi_embedding,
        pvi_null=result.pvi_null,
        pvi_gain=result.pvi_gain,
        mdl_embedding_bits=result.mdl_embedding_bits,
        mdl_null_bits=result.mdl_null_bits,
        min_pvi_gain=result.min_pvi_gain,
        passes=result.passes,
    )
    assert check.passes == result.passes


def test_assess_triviality_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        assess_triviality(
            np.zeros((10, 3)),
            np.zeros((10, 1)),
            np.zeros(9),
            task_type="binary",
            null_input="zeros",
        )
