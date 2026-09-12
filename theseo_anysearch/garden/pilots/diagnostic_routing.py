"""Development-only experiment routing, separate from historical promotion gates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Outcome(str, Enum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class DiagnosticEvidence:
    """PASS/FAIL must come from the frozen diagnostic criteria, not old gates."""

    validity_errors: tuple[str, ...] = ()
    elapsed_hours: float = 0.0
    cap_hours: float = 2.0
    d0: Outcome = Outcome.PENDING
    d1: Outcome = Outcome.PENDING
    d2: Outcome = Outcome.PENDING
    d3: Outcome = Outcome.PENDING
    d4: Outcome = Outcome.PENDING

    def __post_init__(self) -> None:
        for name in ("elapsed_hours", "cap_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric hours")
        if self.elapsed_hours < 0 or self.cap_hours <= 0:
            raise ValueError("elapsed hours must be nonnegative and cap positive")
        if not isinstance(self.validity_errors, tuple) or any(
            not isinstance(v, str) or not v.strip() for v in self.validity_errors
        ):
            raise ValueError("validity_errors must be a tuple of nonempty reasons")
        for name in ("d0", "d1", "d2", "d3", "d4"):
            if not isinstance(getattr(self, name), Outcome):
                raise ValueError(f"{name} must be an Outcome")
        if self.d0 != Outcome.PASS and any(
            getattr(self, k) != Outcome.PENDING for k in ("d1", "d2", "d3", "d4")
        ):
            raise ValueError("D1-D4 require successful D0")
        if self.d3 != Outcome.PENDING and not (
            self.d1 == Outcome.FAIL and self.d2 in (Outcome.PASS, Outcome.FAIL)
        ):
            raise ValueError("D3 requires a failed visible-input reference")
        if self.d4 != Outcome.PENDING and not (self.d1 == Outcome.PASS and self.d2 == Outcome.FAIL):
            raise ValueError("D4 requires successful reference and failed probe ladder")


@dataclass(frozen=True)
class DiagnosticDecision:
    validity: str
    outcome: str
    next_experiments: tuple[str, ...]
    interpretation: str
    promotion_eligible: bool = False


def route(e: DiagnosticEvidence) -> DiagnosticDecision:
    """Choose bounded follow-ups; this function never authorizes promotion."""
    def decision(outcome: str, next_steps: tuple[str, ...], reason: str) -> DiagnosticDecision:
        return DiagnosticDecision("valid", outcome, next_steps, reason)

    if e.validity_errors:
        return DiagnosticDecision("invalid", "repair_required", (), "; ".join(e.validity_errors))
    if e.d0 in (Outcome.FAIL, Outcome.INCONCLUSIVE):
        return decision("repair_required", (), "Pipeline sanity is unresolved; inspect fitting, labels and metrics.")
    # Completed diagnostics are assessable at the cap; only further work is prohibited.
    if e.d1 == Outcome.PASS and e.d2 == Outcome.PASS:
        return decision("readout_recovery", (), "The registered probe ladder recovered the target; no general encoder certification follows.")
    if e.d3 in (Outcome.PASS, Outcome.FAIL):
        return decision("observation_sensitive" if e.d3 == Outcome.PASS else "inconclusive", (),
                        "Observation controls improved recovery." if e.d3 == Outcome.PASS else
                        "Reference and observation controls did not establish headroom; impossibility is not proven.")
    if e.d4 in (Outcome.PASS, Outcome.FAIL):
        return decision("coverage_sensitive" if e.d4 == Outcome.PASS else "representation_unresolved", (),
                        "Broader training coverage improved recovery." if e.d4 == Outcome.PASS else
                        "Coverage control did not resolve the gap; a separately scoped architecture study may follow.")
    if any(getattr(e, k) == Outcome.INCONCLUSIVE for k in ("d1", "d2", "d3", "d4")):
        return decision("inconclusive", (), "Diagnostic uncertainty remains; no automatic retries or budget extension.")
    if e.elapsed_hours >= e.cap_hours:
        return decision("budget_exhausted", (), "Assess available evidence; do not launch another fit.")
    if e.d0 == Outcome.PENDING:
        return decision("ready", ("D0",), "Verify the pipeline on a tiny fixture before comparisons.")
    pending = tuple(k.upper() for k in ("d1", "d2") if getattr(e, k) == Outcome.PENDING)
    if pending:
        return decision("ready", pending, "Reference and frozen-feature ladder can run independently after D0.")
    if e.d1 == Outcome.FAIL:
        return decision("reference_unresolved", ("D3",), "Check observation and resolution; failed learning does not prove unidentifiability.")
    return decision("representation_suspected", ("D4",), "Reference succeeded but registered probes failed; test training coverage next.")
