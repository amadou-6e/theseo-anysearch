"""Pre-data tests for the registered R0 runner; no audit identities are opened."""
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from experiments.perception_encoder.v2r2_r0 import resolve_protocol, select_quota, write_json_once, _initialize_worker, _build
from theseo_anysearch.garden.pilots.io import read_contract, write_contract
from theseo_anysearch.garden.pilots.v2r2_data import (
    build_context, completed_draws, identity_plan, nearest_visible_donors, observed_query,
)
from theseo_anysearch.garden.pilots.v2r2_protocol import V2R2AuditProtocol, V2R2ComparativeContract, comparative_from_reports
from theseo_anysearch.garden.pilots.v2r2_controls import example_from_completions


def config():
    return yaml.safe_load(Path("experiments/perception_encoder/v2r2-r0-config.yaml").read_text())


def protocol():
    return resolve_protocol(config(), executable_sha="a" * 40, spec_sha="b" * 40)


def test_complete_registration_roundtrip(tmp_path):
    registered = protocol()
    path = tmp_path / "protocol.json"
    write_contract(path, registered)
    assert read_contract(path, V2R2AuditProtocol) == registered
    assert len(registered.pools["r0_in_domain"]) == 144
    identities = [row["geometry_id"] for pool in registered.pools.values() for row in pool]
    assert len(set(identities)) == len(identities)
    assert all(name.startswith("pilot-v2r2-") for name in identities)


@pytest.mark.parametrize("field", ["calibration", "controls", "assessment", "r0_wall_cap_seconds", "required_topology_components"])
def test_missing_registration_sections_cannot_freeze(field):
    values = config()
    del values[field]
    with pytest.raises((ValidationError, KeyError)):
        resolve_protocol(values, executable_sha="a" * 40, spec_sha="b" * 40)


def test_pool_hash_and_topology_cannot_be_changed():
    values = protocol().model_dump()
    values["pool_plan_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="pool identity"):
        V2R2AuditProtocol(**values)
    values = protocol().model_dump()
    values["required_topology_components"] = ()
    with pytest.raises(ValidationError, match="topology"):
        V2R2AuditProtocol(**values)


def test_comparative_contract_rejects_deferred_audit():
    with pytest.raises(ValidationError):
        V2R2ComparativeContract(r0_decision="defer")


def arrays():
    observed = np.zeros((17, 17, 17), dtype=bool)
    observed[1, 1, 1] = True
    unknown = np.zeros_like(observed)
    unknown[7:10] = True
    donors = np.stack([observed] * 4)
    donors[2:, 8] = True
    return observed, unknown, donors


def test_hidden_donor_values_cannot_affect_conditioning_rank():
    observed, unknown, donors = arrays()
    expected = nearest_visible_donors(observed, unknown, donors, 4)
    donors[:, unknown] = ~donors[:, unknown]
    np.testing.assert_array_equal(nearest_visible_donors(observed, unknown, donors, 4), expected)


def test_iid_draws_preserve_observed_cells_and_repeat_by_seed():
    observed, unknown, donors = arrays()
    completed, selected, choices = completed_draws(observed, unknown, donors, k=4, count=64, seed=123)
    assert np.all(completed[:, ~unknown] == observed[~unknown])
    assert len(set(choices)) == 4
    np.testing.assert_array_equal(completed_draws(observed, unknown, donors, k=4, count=64, seed=123)[0], completed)


def test_query_is_visible_and_span_is_not_conditioned_on_label():
    observed, unknown, _ = arrays()
    start, goal, span = observed_query(observed, unknown, seed=1, stratum="3-5", attempts=1)
    assert span == 3
    assert start[1:] == goal[1:]
    assert not unknown[start] and not unknown[goal]


def test_native_context_builder_with_development_volume(monkeypatch):
    observed, unknown, donors = arrays()
    monkeypatch.setattr("theseo_anysearch.garden.pilots.v2r2_data.native_volume", lambda *args: observed)
    record = {"geometry_id": "development-only", "stratum": "3-5", "configuration": "A", "family": "topology", "occupancy_band": "low"}
    row, audit = build_context(record, donors, seed=123, k=4, completions=64, query_attempts=1)
    assert row.geometry_id == "development-only"
    assert set(row.labels) == {0, 1}
    assert audit["observed_span"] == 3


def test_quota_uses_no_labels_and_prevents_observation_reuse():
    observed, unknown, donors = arrays()
    row = example_from_completions(context_id="dev", geometry_id="dev", generator_configuration="A",
        bootstrap_stratum="topology:low", stratum="3-5", observed_occupancy=observed, unknown=unknown,
        start=(6, 0, 0), goal=(10, 0, 0), completions=donors, forbidden_features=(1.0,))
    changed = replace(row, labels=(1, 1, 1, 1))
    assert select_quota([(row, {})], 12, set())[0][0].geometry_id == select_quota([(changed, {})], 12, set())[0][0].geometry_id
    selected, exclusions = select_quota([(row, {})], 12, {row.observation_sha256})
    assert not selected
    assert exclusions["dev"] == "duplicate_selected_observation"


def test_reports_are_write_once(tmp_path):
    path = tmp_path / "report.json"
    write_json_once(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"status": "replacement"})


def test_comparative_evidence_tampering_cannot_freeze():
    with pytest.raises(ValueError, match="hash mismatch"):
        comparative_from_reports(protocol(), {"status": "feasible"}, {})


def test_windows_worker_path_with_development_only_native_ids():
    _, _, donors = arrays()
    records = [{"geometry_id": "development-native-r0-worker-0", "family": "open", "occupancy_band": "low",
        "source": "development_only", "parent_split": "train", "confirmation_group": None,
        "configuration": "A", "stratum": "3-5"}]
    settings = {"seed": 1, "k": 4, "completions": 8, "query_attempts": 1}
    with ProcessPoolExecutor(max_workers=1, initializer=_initialize_worker, initargs=({"A": donors}, settings)) as executor:
        result = list(executor.map(_build, records))
    assert result[0][1]["geometry_id"].startswith("development-")
    assert result[0][1]["status"] in {"generated", "no_observed_query", "no_visible_occupancy"}
