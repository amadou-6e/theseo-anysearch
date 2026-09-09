import copy
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
