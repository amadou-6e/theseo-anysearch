import pytest
import json
from pathlib import Path
from theseo_anysearch.garden.pilots import distance_retention as d


@pytest.mark.parametrize("new,expected",[(.05,"distance_retained"),(.10,"distance_retention_unconfirmed")])
def test_retention_gate(new,expected):
    trials=[{"seed":s,"arm":arm,"task":task,"nmae":v,"geometry_nmae":[v]*48}
            for s in range(3) for arm,v in (("old",.04),("new",new)) for task in d.PLAN["tasks"]]
    assert d.assess(trials)["outcome"]==expected


def test_missing_seeds():
    assert d.assess([])["outcome"]=="inconclusive"


def test_evidence_replay():
    root=Path(__file__).resolve().parents[2]/"docs/perception-encoder-local-geometry"
    report=json.loads((root/"retention-report.json").read_text())
    sha=report.pop("report_payload_sha256")
    assert d.d.base.payload_sha256(report)==sha
    assert d.assess(report["trials"])==report["assessment"]
    env=json.loads((root/"retention-preregistration.json").read_text())
    assert report["registration"]==env
    assert d.d.base.payload_sha256(env["payload"])==env["identity_sha256"]
    prior=json.loads((root/"continuation-report.json").read_text())
    sha=prior.pop("report_payload_sha256")
    assert d.d.base.payload_sha256(prior)==sha
    assert d.e.assess(prior["trials"])==prior["assessment"]
