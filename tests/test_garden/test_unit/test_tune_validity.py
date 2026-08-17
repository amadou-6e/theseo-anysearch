from theseo_anysearch.garden.tune_runner import _validity_metrics


def test_validity_metrics_rank_valid_trial_above_invalid_trial():
    gates = {"probe_iou": 0.8, "probe_ood_iou": 0.65}
    valid = _validity_metrics(
        {"representation_score": 0.82, "probe_iou": 0.81, "probe_ood_iou": 0.66},
        gates,
    )
    invalid = _validity_metrics(
        {"representation_score": 0.90, "probe_iou": 0.90, "probe_ood_iou": 0.60},
        gates,
    )
    assert valid["valid_trial"] == 1.0
    assert invalid["valid_trial"] == 0.0
    assert valid["valid_representation_score"] > invalid["valid_representation_score"]


def test_invalid_trials_retain_a_smooth_pruning_signal():
    gates = {"probe_iou": 0.8, "probe_ood_iou": 0.65}
    closer = _validity_metrics(
        {"representation_score": 0.8, "probe_iou": 0.78, "probe_ood_iou": 0.64}, gates
    )
    farther = _validity_metrics(
        {"representation_score": 0.8, "probe_iou": 0.60, "probe_ood_iou": 0.50}, gates
    )
    assert closer["valid_representation_score"] > farther["valid_representation_score"]
