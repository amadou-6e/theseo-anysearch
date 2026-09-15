"""Fixed Bayesian learning-curve extrapolation with a mandatory backtest gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


CURVE_IMPLEMENTATION_VERSION = "bayesian-inverse-power-v1"


@dataclass(frozen=True)
class CurvePrior:
    mean: tuple[float, float] = (0.5, 0.0)
    standard_deviation: tuple[float, float] = (0.5, 2.0)
    observation_noise: float = 0.05


@dataclass(frozen=True)
class CurvePrediction:
    target_update: int
    mean: float
    lower_95: float
    upper_95: float
    posterior_draws: np.ndarray
    implementation_version: str = CURVE_IMPLEMENTATION_VERSION


@dataclass(frozen=True)
class CurveCalibration:
    coverage_95: float
    median_absolute_error: float
    hidden_points: int
    calibrated: bool


def target_horizon(current_budget: int, next_stage_budget: int) -> int:
    if current_budget <= 0 or next_stage_budget <= 0:
        raise ValueError("learning-curve budgets must be positive")
    return min(4 * current_budget, next_stage_budget)


def _design(updates: np.ndarray) -> np.ndarray:
    if np.any(updates <= 0):
        raise ValueError("learning-curve update positions must be positive")
    return np.column_stack((np.ones_like(updates), 1.0 / np.sqrt(updates)))


def predict_learning_curve(
    updates: Sequence[int],
    scores: Sequence[float],
    *,
    target_update: int,
    prior: CurvePrior = CurvePrior(),
    posterior_draws: int = 4_096,
    seed: int = 0,
) -> CurvePrediction:
    """Bayesian linear regression over score = asymptote + slope/sqrt(update)."""

    x = np.asarray(updates, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    if len(x) != len(y) or len(x) < 3 or not np.isfinite(y).all():
        raise ValueError("curve fitting requires at least three aligned finite observations")
    if np.any(np.diff(x) <= 0) or target_update < x[-1]:
        raise ValueError("updates must increase and target cannot precede observations")
    if prior.observation_noise <= 0 or any(value <= 0 for value in prior.standard_deviation):
        raise ValueError("curve prior scales must be positive")
    if posterior_draws < 100:
        raise ValueError("at least 100 posterior draws are required")

    design = _design(x)
    prior_mean = np.asarray(prior.mean)
    prior_precision = np.diag(1.0 / np.square(prior.standard_deviation))
    noise_variance = prior.observation_noise**2
    posterior_precision = prior_precision + design.T @ design / noise_variance
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ (
        prior_precision @ prior_mean + design.T @ y / noise_variance
    )
    target_design = _design(np.asarray([target_update], dtype=np.float64))[0]
    rng = np.random.default_rng(seed)
    coefficients = rng.multivariate_normal(
        posterior_mean, posterior_covariance, size=posterior_draws
    )
    draws = coefficients @ target_design
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return CurvePrediction(
        target_update=target_update,
        mean=float(draws.mean()),
        lower_95=float(lower),
        upper_95=float(upper),
        posterior_draws=draws,
    )


def backtest_learning_curves(
    curves: Sequence[tuple[Sequence[int], Sequence[float]]],
    *,
    prior: CurvePrior = CurvePrior(),
    seed: int = 0,
) -> CurveCalibration:
    """Hide each curve's final two points and apply the preregistered calibration gate."""

    covered: list[bool] = []
    errors: list[float] = []
    for curve_index, (updates, scores) in enumerate(curves):
        if len(updates) != len(scores) or len(updates) < 5:
            raise ValueError("backtest curves require at least five aligned checkpoints")
        for hidden_index in (-2, -1):
            prediction = predict_learning_curve(
                updates[:-2],
                scores[:-2],
                target_update=int(updates[hidden_index]),
                prior=prior,
                seed=seed + curve_index * 2 + (hidden_index + 2),
            )
            actual = float(scores[hidden_index])
            covered.append(prediction.lower_95 <= actual <= prediction.upper_95)
            errors.append(abs(prediction.mean - actual))
    coverage = float(np.mean(covered))
    median_error = float(np.median(errors))
    return CurveCalibration(
        coverage_95=coverage,
        median_absolute_error=median_error,
        hidden_points=len(errors),
        calibrated=coverage >= 0.90 and median_error <= 0.05,
    )


def extrapolation_can_rescue(candidate: CurvePrediction, leader: CurvePrediction) -> bool:
    """Extrapolation may retain, but never select, when posterior intervals overlap."""

    return candidate.upper_95 >= leader.lower_95
