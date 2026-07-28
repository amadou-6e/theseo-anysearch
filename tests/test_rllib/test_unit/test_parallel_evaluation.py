from types import SimpleNamespace

from theseo_anysearch.rllib.trainer.parallel_evaluation import (
    collect_rllib_evaluation_episodes,
    configure_rllib_evaluation,
)


class _FakeEvaluationGroup:
    def __init__(self) -> None:
        self.synced_from = None
        self.worker_ids = [11, 12, 13]
        self.received_worker_ids = None
        self.local_env_runner = None

    def healthy_env_runner_ids(self):
        return self.worker_ids

    def sync_weights(self, *, from_worker_or_learner_group, inference_only):
        self.synced_from = from_worker_or_learner_group
        assert inference_only is True

    def foreach_env_runner(
        self,
        functions,
        *,
        local_env_runner,
        remote_worker_ids,
    ):
        assert local_env_runner is False
        self.received_worker_ids = remote_worker_ids
        results = []
        for function in reversed(functions):
            results.append([
                (seed, f"episode-{seed}")
                for seed in function.keywords["seeds"]
            ])
        return results


def test_parallel_collection_distributes_and_restores_seed_order() -> None:
    group = _FakeEvaluationGroup()
    source = object()
    algorithm = SimpleNamespace(
        eval_env_runner_group=group,
        env_runner=source,
    )

    episodes = collect_rllib_evaluation_episodes(
        algorithm,
        {},
        5,
        seed=42,
        multi_agent=False,
    )

    assert episodes == [
        "episode-42",
        "episode-43",
        "episode-44",
        "episode-45",
        "episode-46",
    ]
    assert group.synced_from is source
    assert group.received_worker_ids == [11, 12, 13]


def test_parallel_collection_uses_only_workers_with_assigned_episodes() -> None:
    group = _FakeEvaluationGroup()
    algorithm = SimpleNamespace(
        eval_env_runner_group=group,
        env_runner=object(),
    )

    episodes = collect_rllib_evaluation_episodes(
        algorithm,
        {},
        2,
        seed=7,
        multi_agent=False,
    )

    assert episodes == ["episode-7", "episode-8"]
    assert group.received_worker_ids == [11, 12]


class _FakeRllibConfig:
    def __init__(self) -> None:
        self.options = None

    def evaluation(self, **options):
        self.options = options
        return self


def test_rllib_evaluation_configuration_creates_dedicated_workers() -> None:
    config = _FakeRllibConfig()

    result = configure_rllib_evaluation(config, num_env_runners=8)

    assert result is config
    assert config.options == {
        "evaluation_interval": None,
        "evaluation_num_env_runners": 8,
        "evaluation_parallel_to_training": False,
        "evaluation_config": {"explore": False},
    }
