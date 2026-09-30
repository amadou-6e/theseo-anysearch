"""Contracts for canonical task and optimization metric names."""

from theseo_anysearch.rllib.trainer.reporting.metrics import canonical_rllib_metrics


def test_available_optimizer_metrics_are_mapped_without_empty_algorithm_series():
    metrics = canonical_rllib_metrics({
        "learners": {
            "default_policy": {
                "policy_loss": -0.2,
                "vf_loss": 0.4,
                "entropy": 0.7,
                "mean_kl_loss": 0.01,
            }
        }
    })

    assert metrics == {
        "train/optimization/policy_loss": -0.2,
        "train/optimization/value_loss": 0.4,
        "train/optimization/entropy": 0.7,
        "train/optimization/approx_kl": 0.01,
    }
    assert not any("td_loss" in name for name in metrics)


def test_task_metrics_are_read_only_from_training_env_runner_results():
    metrics = canonical_rllib_metrics({
        "env_runners": {
            "success_rate": 0.5,
            "waypoint_completion_fraction_mean": 0.75,
            "collision_rate": 0.1,
        },
        "evaluation": {"success_rate": 1.0},
    })

    assert metrics == {
        "train/task/success_rate": 0.5,
        "train/task/waypoint/completion_fraction_mean": 0.75,
        "train/task/collision_rate": 0.1,
    }
