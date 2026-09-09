import copy
import json
from pathlib import Path
import pytest
from theseo_anysearch.garden.pilots import checkpoint_diagnostic as c


@pytest.mark.parametrize("best,final,probe,outcome", [
    (.2,.5,.3,"reference_exceeds_probe"),
    (.3,.5,.2,"reference_improves_below_probe"),
    (.5,.5,.2,"selection_benefit_unresolved"),
    (.3,.5,.3,"reference_probe_difference_unresolved"),
])
def test_paired_routing(best, final, probe, outcome):
    trials = [{"seed": s, **{k: {"geometry_log_loss": [v]*48}
               for k,v in (("best",best),("final",final),("probe",probe))}} for s in range(3)]
    assert c.assess(trials)["outcome"] == outcome
    assert not c.assess(trials)["promotion_eligible"]


def test_incomplete_seeds():
    assert c.assess([])["outcome"] == "inconclusive"


def test_corpus_configuration_does_not_mutate_old_plan():
    old = copy.deepcopy(c.d.PLAN)
    assert c.PLAN["dataset_id"] != old["dataset_id"]
    assert c.PLAN["streams"] != old["streams"]
    assert c.d.PLAN == old


def test_report_replay_and_selection():
    root = Path(__file__).resolve().parents[2] / "docs/perception-encoder-local-geometry"
    r = json.loads((root / "checkpoint-report.json").read_text())
    digest = r.pop("report_payload_sha256")
    assert c.d.base.payload_sha256(r) == digest
    assert c.assess(r["trials"]) == r["assessment"]
    envelope = json.loads((root / "checkpoint-preregistration.json").read_text())
    assert r["registration"] == envelope
    assert c.d.base.payload_sha256(envelope["payload"]) == envelope["identity_sha256"]
    for t in r["trials"]:
        assert t["best_step"] == min(t["curve"], key=lambda x: x["selection_loss"])["step"]
        selected = min(t["capacities"], key=lambda x: x["selection_loss"])
        assert t["selected_width"] == selected["width"]
        assert t["probe"] == selected["assessment"]
        assert t["encoder_state_before"] == t["encoder_state_after"]
