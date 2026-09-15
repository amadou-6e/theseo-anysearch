from theseo_anysearch.garden.pilots import encoder_continuation as e


def test_matched_exposure_and_retraining_direction():
    trials=[{"seed":s,**{k:{"assessment":{"geometry_log_loss":[v]*48}}
             for k,v in (("old48",.3),("old192",.25),("new192",.2),("reference",.15))}} for s in range(3)]
    result=e.assess(trials)
    assert result["exposure_loss_gain"]["ci95"][0]>0
    assert result["retraining_loss_gain"]["ci95"][0]>0
    assert result["next_action"]=="validate_distance_retention_and_new_families"
    assert not result["promotion_eligible"]


def test_incomplete_does_not_select_direction():
    assert e.assess([])["outcome"]=="inconclusive"
