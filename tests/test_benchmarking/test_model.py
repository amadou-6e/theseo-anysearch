"""Tests for the roofline predictive model in theseo_anysearch.benchmarking.model."""

from __future__ import annotations

import math

import pytest

from theseo_anysearch.benchmarking.model import (
    StageCosts,
    fit_contention_correction,
    predict_throughput,
    recommend_search_range,
)


def _costs(**overrides: object) -> StageCosts:
    defaults: dict[str, object] = dict(
        env_step_seconds=0.01,
        inference_seconds_per_env=0.005,
        transfer_seconds_per_mb=0.002,
        avg_sample_mb=1.0,
        learner_seconds_per_batch=0.5,
        train_batch_size=1000,
    )
    defaults.update(overrides)
    return StageCosts(**defaults)


class TestPredictThroughput:

    def test_rejects_non_positive_candidates(self) -> None:
        costs = _costs()
        with pytest.raises(ValueError):
            predict_throughput(costs, 0, 1)
        with pytest.raises(ValueError):
            predict_throughput(costs, 1, 0)

    def test_learner_is_bottleneck_when_gpu_is_slow(self) -> None:
        costs = _costs(learner_seconds_per_batch=1000.0)
        predicted = predict_throughput(costs, 4, 4)
        assert predicted.bottleneck == "learner"
        assert predicted.predicted_steps_per_second == pytest.approx(1000 / 1000.0)

    def test_env_step_is_bottleneck_when_stepping_is_slow(self) -> None:
        costs = _costs(
            env_step_seconds=100.0,
            inference_seconds_per_env=1e-9,
            learner_seconds_per_batch=0.001,
        )
        predicted = predict_throughput(costs, 1, 1)
        assert predicted.bottleneck == "env_step"
        assert predicted.predicted_steps_per_second == pytest.approx(1 / 100.0, rel=1e-6)

    def test_gil_contention_reduces_rate_and_can_become_bottleneck(self) -> None:
        low_contention = _costs(gil_contention_ratio=0.0, learner_seconds_per_batch=0.001)
        high_contention = _costs(gil_contention_ratio=0.9, learner_seconds_per_batch=0.001)
        low = predict_throughput(low_contention, 1, 1)
        high = predict_throughput(high_contention, 1, 1)
        assert high.predicted_steps_per_second < low.predicted_steps_per_second
        assert high.bottleneck == "gil"

    def test_gil_none_does_not_constrain_the_model(self) -> None:
        costs = _costs(gil_contention_ratio=None, learner_seconds_per_batch=0.001)
        predicted = predict_throughput(costs, 1, 1)
        assert predicted.bottleneck != "gil"

    def test_more_env_runners_increases_predicted_rate_without_correction(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        one = predict_throughput(costs, 1, 1)
        four = predict_throughput(costs, 4, 1)
        assert four.predicted_steps_per_second == pytest.approx(
            4 * one.predicted_steps_per_second)

    def test_correction_penalizes_higher_env_runner_counts(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        uncorrected = predict_throughput(costs, 8, 1, correction=0.0)
        corrected = predict_throughput(costs, 8, 1, correction=0.5)
        assert corrected.predicted_steps_per_second < uncorrected.predicted_steps_per_second

    def test_correction_does_not_affect_single_env_runner(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        uncorrected = predict_throughput(costs, 1, 1, correction=0.0)
        corrected = predict_throughput(costs, 1, 1, correction=0.9)
        assert corrected.predicted_steps_per_second == pytest.approx(
            uncorrected.predicted_steps_per_second)

    def test_scheduler_queue_delay_can_become_bottleneck(self) -> None:
        costs = _costs(
            learner_seconds_per_batch=1e-6,
            scheduler_queue_seconds=1000.0,
        )
        predicted = predict_throughput(costs, 8, 1)
        assert predicted.bottleneck == "scheduler"


class TestFitContentionCorrection:

    def test_returns_zero_with_no_multi_runner_probes(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        assert fit_contention_correction(costs, [(1, 1, 100.0)]) == 0.0

    def test_returns_zero_when_measurements_match_naive_prediction(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        naive_at_4 = predict_throughput(costs, 4, 1).predicted_steps_per_second
        naive_at_8 = predict_throughput(costs, 8, 1).predicted_steps_per_second
        correction = fit_contention_correction(
            costs, [(4, 1, naive_at_4), (8, 1, naive_at_8)])
        assert correction == pytest.approx(0.0, abs=1e-6)

    def test_recovers_known_penalty_exponent(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        true_exponent = 0.3
        probe_points = []
        for n in (2, 4, 8):
            naive = predict_throughput(costs, n, 1).predicted_steps_per_second
            measured = naive * n**(-true_exponent)
            probe_points.append((n, 1, measured))

        fitted = fit_contention_correction(costs, probe_points)
        assert fitted == pytest.approx(true_exponent, abs=1e-6)

    def test_ignores_zero_or_single_runner_probes(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        naive_at_4 = predict_throughput(costs, 4, 1).predicted_steps_per_second
        with_noise = fit_contention_correction(
            costs, [(1, 1, 999.0), (4, 1, naive_at_4)])
        without_noise = fit_contention_correction(costs, [(4, 1, naive_at_4)])
        assert with_noise == pytest.approx(without_noise)

    def test_clamps_to_0_6_even_for_an_extreme_single_noisy_probe(self) -> None:
        # A single probe implying near-total collapse (e.g. a one-off stall)
        # must not be allowed to zero out all predicted parallelism benefit.
        costs = _costs(learner_seconds_per_batch=1e-6)
        naive_at_2 = predict_throughput(costs, 2, 1).predicted_steps_per_second
        fitted = fit_contention_correction(costs, [(2, 1, naive_at_2 * 0.001)])
        assert fitted == pytest.approx(0.6)


class TestRecommendSearchRange:

    def test_rejects_invalid_maximum(self) -> None:
        costs = _costs()
        with pytest.raises(ValueError):
            recommend_search_range(costs, axis="num_env_runners", maximum=0)

    def test_range_stays_within_bounds(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        start, end = recommend_search_range(
            costs, axis="num_env_runners", maximum=10, band=2)
        assert 1 <= start <= end <= 10

    def test_band_centers_on_predicted_peak_when_sampling_bound_throughout(self) -> None:
        # Fast learner and fast transfer mean sampling never saturates either
        # downstream stage -> more env-runners always helps -> peak sits at maximum.
        costs = _costs(learner_seconds_per_batch=1e-6, transfer_seconds_per_mb=1e-9)
        start, end = recommend_search_range(
            costs, axis="num_env_runners", maximum=10, band=2)
        assert end == 10
        assert start == 8

    def test_band_centers_on_predicted_peak_when_gpu_saturates_early(self) -> None:
        # A slow learner means throughput plateaus once sampling outpaces it;
        # extra env-runners beyond that point cannot raise the roofline
        # prediction further, so the peak should land near where sampling
        # first reaches the learner's rate rather than at the maximum.
        costs = _costs(env_step_seconds=1.0, inference_seconds_per_env=1e-9,
                        learner_seconds_per_batch=2.0, train_batch_size=1)
        start, end = recommend_search_range(
            costs, axis="num_env_runners", maximum=10, band=1)
        assert end < 10

    def test_axis_selects_which_dimension_varies(self) -> None:
        costs = _costs(learner_seconds_per_batch=1e-6)
        runners_start, runners_end = recommend_search_range(
            costs, axis="num_env_runners", maximum=10, fixed=1, band=1)
        envs_start, envs_end = recommend_search_range(
            costs, axis="num_envs_per_env_runner", maximum=10, fixed=1, band=1)
        # With symmetric costs (inference/env == step scaling), both axes
        # should behave identically since total_envs = runners * envs_per_runner.
        assert (runners_start, runners_end) == (envs_start, envs_end)
