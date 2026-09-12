import pytest

from theseo_anysearch.garden.pilots.diagnostic_routing import DiagnosticEvidence as E, Outcome as O, route


@pytest.mark.parametrize("e,expected,steps", [
    (E(), "ready", ("D0",)),
    (E(d0=O.FAIL), "repair_required", ()),
    (E(d0=O.PASS), "ready", ("D1", "D2")),
    (E(d0=O.PASS, d1=O.PASS), "ready", ("D2",)),
    (E(d0=O.PASS, d1=O.FAIL, d2=O.FAIL), "reference_unresolved", ("D3",)),
    (E(d0=O.PASS, d1=O.PASS, d2=O.FAIL), "representation_suspected", ("D4",)),
    (E(d0=O.PASS, d1=O.PASS, d2=O.PASS), "readout_recovery", ()),
    (E(d0=O.PASS, d1=O.FAIL, d2=O.PASS), "reference_unresolved", ("D3",)),
    (E(d0=O.PASS, d1=O.FAIL, d2=O.FAIL, d3=O.PASS), "observation_sensitive", ()),
    (E(d0=O.PASS, d1=O.FAIL, d2=O.FAIL, d3=O.FAIL), "inconclusive", ()),
    (E(d0=O.PASS, d1=O.PASS, d2=O.FAIL, d4=O.PASS), "coverage_sensitive", ()),
    (E(d0=O.PASS, d1=O.PASS, d2=O.FAIL, d4=O.FAIL), "representation_unresolved", ()),
    (E(d0=O.PASS, d1=O.INCONCLUSIVE), "inconclusive", ()),
    (E(elapsed_hours=2), "budget_exhausted", ()),
    (E(d0=O.PASS, d1=O.PASS, d2=O.PASS, elapsed_hours=2), "readout_recovery", ()),
])
def test_routes(e, expected, steps):
    result = route(e)
    assert result.outcome == expected
    assert result.next_experiments == steps
    assert not result.promotion_eligible


def test_invalid_evidence_overrides_quality_and_budget():
    result = route(E(validity_errors=("leakage",), elapsed_hours=2,
                     d0=O.PASS, d1=O.PASS, d2=O.PASS))
    assert result.validity == "invalid"
    assert result.next_experiments == ()


@pytest.mark.parametrize("kwargs", [
    {"elapsed_hours": float("nan")}, {"cap_hours": float("inf")},
    {"elapsed_hours": -1}, {"cap_hours": 0}, {"cap_hours": True},
    {"d0": "pass"}, {"d1": O.PASS}, {"validity_errors": ("",)},
    {"d0": O.PASS, "d3": O.PASS},
    {"d0": O.PASS, "d1": O.FAIL, "d4": O.PASS},
    {"d0": O.PASS, "d1": O.FAIL, "d3": O.PASS},
])
def test_rejects_malformed_or_impossible_evidence(kwargs):
    with pytest.raises(ValueError):
        E(**kwargs)
