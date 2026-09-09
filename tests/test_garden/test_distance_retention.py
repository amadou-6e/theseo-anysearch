import pytest
from theseo_anysearch.garden.pilots import distance_retention as d


@pytest.mark.parametrize("new,expected",[(.05,"distance_retained"),(.10,"distance_retention_unconfirmed")])
def test_retention_gate(new,expected):
    trials=[{"seed":s,"arm":arm,"task":task,"nmae":v,"geometry_nmae":[v]*48}
            for s in range(3) for arm,v in (("old",.04),("new",new)) for task in d.PLAN["tasks"]]
    assert d.assess(trials)["outcome"]==expected


def test_missing_seeds():
    assert d.assess([])["outcome"]=="inconclusive"
