"""Unit tests for the reachability denominator redesign (R1-R5, #339)."""
from __future__ import annotations

import numpy as np
import pytest

from theseo_anysearch.garden.evaluation.reachability_variants import (
    coordinates_null,
    pair_matrix,
    raw_observed_feature,
    rich_completed_feature,
    sample_occlusion_stratified_pairs,
)
from theseo_anysearch.garden.pilots.reachability_fixtures import (
    generate_occluded_geometry,
    occlusion_corpus,
    occlusion_span_along_path,
)


# --- R1 fixtures ------------------------------------------------------------


def test_corpus_is_deterministic() -> None:
    a = generate_occluded_geometry(7)
    b = generate_occluded_geometry(7)
    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.unknown, b.unknown)


def test_corpus_contains_genuine_disconnection_and_occlusion() -> None:
    corpus = occlusion_corpus(48, seed=20260903)
    component_counts = [int(g.free_component_labels.max()) for g in corpus]
    assert component_counts.count(2) >= 8  # some worlds truly split
    assert all(g.unknown.any() for g in corpus)
    # observed free space never exceeds the completed free space
    for g in corpus[:6]:
        assert np.all(g.observed_free <= g.completed_free)


def test_occlusion_span_is_negative_across_components_and_positive_on_hidden_paths() -> None:
    corpus = occlusion_corpus(20, seed=20260903)
    saw_unreachable = saw_occluded_reachable = False
    for g in corpus:
        sample = sample_occlusion_stratified_pairs(g, count=32, seed=1)
        for reach, span in zip(sample.reachable, sample.occlusion_span):
            if not reach:
                assert span == -1
                saw_unreachable = True
            elif span > 0:
                saw_occluded_reachable = True
    assert saw_unreachable and saw_occluded_reachable


# --- R1-R3 features ------------------------------------------------------------


def test_features_are_finite_fixed_length_and_raw_differs_from_rich() -> None:
    g = generate_occluded_geometry(3)
    coords = np.argwhere(g.completed_free)[:5]
    raw_lens = {len(raw_observed_feature(g, c)) for c in coords}
    rich_lens = {len(rich_completed_feature(g, c)) for c in coords}
    assert len(raw_lens) == 1 and len(rich_lens) == 1
    for c in coords:
        assert np.isfinite(raw_observed_feature(g, c)).all()
        assert np.isfinite(rich_completed_feature(g, c)).all()
        assert len(coordinates_null(g, c)) == 3
    # near the occluded band, observed and completed occupancy disagree
    band = np.argwhere(g.unknown & g.occupancy)
    if len(band):
        c = band[0]
        assert not np.array_equal(
            raw_observed_feature(g, c)[: raw_observed_feature(g, c).size // 2],
            rich_completed_feature(g, c)[: raw_observed_feature(g, c).size // 2],
        )


def test_pair_matrix_shape() -> None:
    g = generate_occluded_geometry(4)
    sample = sample_occlusion_stratified_pairs(g, count=24, seed=2)
    matrix = pair_matrix(raw_observed_feature, g, sample)
    assert matrix.shape[0] == len(sample.starts)
    assert matrix.shape[1] == 3 * len(raw_observed_feature(g, sample.starts[0]))


# --- R5 screen ------------------------------------------------------------


def test_screen_runs_reports_every_variant_and_is_deterministic() -> None:
    from experiments.perception_encoder.reachability_redesign_screen import run_screen

    first = run_screen(n=24, seed=20260903)
    second = run_screen(n=24, seed=20260903)
    assert first["report_payload_sha256"] == second["report_payload_sha256"]
    assert set(first["variants"]) == {
        "V0_pairwise_auprc",
        "A_occlusion_hard_strata",
        "B1_excess_over_raw",
        "B2_pvi_gain",
        "C_component_structure",
    }
    for v in first["variants"].values():
        assert {"floor", "ceiling", "gap", "opens_gate"} <= set(v)
        assert v["gap"] == pytest.approx(v["ceiling"] - v["floor"], abs=1e-9)
    # The screen is non-evidential and cannot recommend adoption.
    assert first["non_evidential"] is True
    assert first["disposition"] == "retain"
    assert first["recommendation"].startswith("non_evidential:")
    assert "adopt:" not in first["recommendation"]


def test_screen_only_records_whether_occlusion_produced_any_headroom() -> None:
    from experiments.perception_encoder.reachability_redesign_screen import run_screen

    report = run_screen(n=48, seed=20260903)
    # Proof-of-mechanism: on an occlusion-rich corpus, occlusion should at least
    # produce *some* headroom in a genuine redesign - but this is not evidence
    # any variant opens the gate on the real corpus.
    genuine = {"A_occlusion_hard_strata", "B2_pvi_gain", "C_component_structure"}
    assert any(report["variants"][name]["opens_gate"] for name in genuine)
    assert report["recommendation"] == (
        "non_evidential:occlusion_creates_headroom:defer_to_v2r2_R0"
    )
