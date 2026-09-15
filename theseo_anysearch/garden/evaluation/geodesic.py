"""Recorded pilot geodesic calibration decision (F5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeodesicPilotDecision:
    """Whether bounded-radius geodesic error contains usable pilot headroom."""

    disposition: str
    frequency_nmae: float
    supervised_nmae: float
    target_standard_deviation: float
    minimum_frequency_nmae: float
    reason: str
    revisit_stage: str

    @property
    def active_in_p0c(self) -> bool:
        return self.disposition == "redesign"


def decide_geodesic_pilot_metric(
    targets: np.ndarray,
    *,
    frequency_nmae: float,
    supervised_nmae: float,
    minimum_frequency_nmae: float = 0.15,
) -> GeodesicPilotDecision:
    """Defer a low-variance metric that a constant baseline already solves.

    Stage 2 supplies the larger physical field of view needed for long-horizon
    geodesic structure. The decision is data-driven and must be recorded rather
    than silently removing a failed denominator.
    """

    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("geodesic decision requires a finite one-dimensional target sample")
    if min(frequency_nmae, supervised_nmae, minimum_frequency_nmae) < 0:
        raise ValueError("geodesic errors and the noise-floor threshold must be nonnegative")
    standard_deviation = float(values.std(ddof=1))
    usable = (
        frequency_nmae >= minimum_frequency_nmae
        and supervised_nmae <= 0.8 * frequency_nmae
    )
    if usable:
        disposition = "redesign"
        reason = (
            "the frequency baseline exceeds the pilot noise floor and supervised "
            "prediction reduces normalized error by at least 20 percent"
        )
    else:
        disposition = "deferred"
        reason = (
            "radius-8/16 geodesic targets have insufficient discriminative headroom: "
            f"frequency NMAE {frequency_nmae:.6f} is below the "
            f"{minimum_frequency_nmae:.2f} noise floor and supervised NMAE "
            f"{supervised_nmae:.6f} does not establish a useful ceiling"
        )
    return GeodesicPilotDecision(
        disposition=disposition,
        frequency_nmae=float(frequency_nmae),
        supervised_nmae=float(supervised_nmae),
        target_standard_deviation=standard_deviation,
        minimum_frequency_nmae=float(minimum_frequency_nmae),
        reason=reason,
        revisit_stage="Stage 2 wide-context evaluation",
    )


__all__ = ["GeodesicPilotDecision", "decide_geodesic_pilot_metric"]
