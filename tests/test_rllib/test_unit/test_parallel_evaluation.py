from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from theseo_anysearch.rllib.trainer.evaluation.parallel import (
    AnySearchEvaluationFunction,
    _PolicyAdapter,
    _collect_worker_episodes,
    _stack_observations,
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
            assert function.keywords["num_envs_per_env_runner"] == 1
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


def test_parallel_collection_syncs_modern_stack_from_learner_group() -> None:
    group = _FakeEvaluationGroup()
    learner_group = object()
    algorithm = SimpleNamespace(
        config=SimpleNamespace(enable_rl_module_and_learner=True),
        eval_env_runner_group=group,
        learner_group=learner_group,
    )

    collect_rllib_evaluation_episodes(
        algorithm,
        {},
        1,
        seed=42,
        multi_agent=False,
    )

    assert group.synced_from is learner_group

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


def test_worker_collection_chunks_vectorized_episodes(monkeypatch) -> None:
    calls = []

    def collect_vectorized(policy, env_config, seeds):
        calls.append((policy, env_config, seeds))
        return [f"episode-{seed}" for seed in seeds]

    monkeypatch.setattr(
        "theseo_anysearch.experiments.trajectory.collect_vectorized_eval_episodes",
        collect_vectorized,
    )
    env_runner = object()

    episodes = _collect_worker_episodes(
        env_runner,
        seeds=(10, 11, 12, 13, 14),
        env_config={"max_steps": 5},
        multi_agent=False,
        num_envs_per_env_runner=2,
    )

    assert [call[2] for call in calls] == [(10, 11), (12, 13), (14,)]
    assert episodes == [
        (10, "episode-10"),
        (11, "episode-11"),
        (12, "episode-12"),
        (13, "episode-13"),
        (14, "episode-14"),
    ]


def test_inline_collection_honors_vectorization(monkeypatch) -> None:
    calls = []

    def collect_vectorized(algorithm, env_config, seeds):
        calls.append(seeds)
        return [f"episode-{seed}" for seed in seeds]

    monkeypatch.setattr(
        "theseo_anysearch.experiments.trajectory.collect_vectorized_eval_episodes",
        collect_vectorized,
    )

    episodes = collect_rllib_evaluation_episodes(
        SimpleNamespace(),
        {},
        5,
        seed=20,
        multi_agent=False,
        num_envs_per_env_runner=2,
    )

    assert calls == [(20, 21), (22, 23), (24,)]
    assert episodes == [
        "episode-20",
        "episode-21",
        "episode-22",
        "episode-23",
        "episode-24",
    ]


def test_stack_observations_preserves_nested_structure() -> None:
    stacked = _stack_observations([
        {"box": [1, 2], "goal": (3, 4)},
        {"box": [5, 6], "goal": (7, 8)},
    ])

    assert stacked["box"].tolist() == [[1, 2], [5, 6]]
    assert [part.tolist() for part in stacked["goal"]] == [[3, 7], [4, 8]]


def test_policy_adapter_flattens_with_explicit_evaluation_space() -> None:
    observation_space = spaces.Dict(
        {
            "box": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            "steps": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
    )
    adapter = _PolicyAdapter(SimpleNamespace(), observation_space=observation_space)

    observations = adapter._module_observations(
        [{"box": np.asarray([0.2, 0.4]), "steps": np.asarray([0.8])}]
    )

    assert observations[0].shape == (3,)
    assert np.allclose(observations[0], [0.2, 0.4, 0.8])


def test_policy_adapter_batches_observations_for_policy(monkeypatch) -> None:
    received = {}

    class _Policy:
        def compute_actions(self, observations, *, explore):
            received["observations"] = observations
            received["explore"] = explore
            return np.asarray([2, 3]), [], {}

    policy = _Policy()
    env_runner = SimpleNamespace(get_policy=lambda policy_id: policy)
    monkeypatch.setattr(
        _PolicyAdapter,
        "_preprocess_observation",
        staticmethod(
            lambda selected_policy, observation, env_id: observation
        ),
    )

    actions, _, _ = _PolicyAdapter(env_runner).compute_actions(
        [{"box": [1, 2]}, {"box": [3, 4]}],
        explore=False,
    )

    assert actions.tolist() == [2, 3]
    assert received["observations"]["box"].tolist() == [[1, 2], [3, 4]]
    assert received["explore"] is False


def test_policy_adapter_uses_policy_for_legacy_rollout_worker() -> None:
    class _LegacyAlgorithm:
        env_runner = SimpleNamespace()

        def get_module(self):
            raise AssertionError("legacy RolloutWorker has no RLModule")

    assert _PolicyAdapter(_LegacyAlgorithm())._rl_module() is None


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
    assert config.options["evaluation_interval"] == 1
    assert config.options["evaluation_num_env_runners"] == 8
    assert config.options["evaluation_parallel_to_training"] is False
    assert config.options["evaluation_config"] == {"explore": False}
    assert isinstance(
        config.options["custom_evaluation_function"],
        AnySearchEvaluationFunction,
    )


def test_rllib_parallel_evaluation_uses_native_scheduler() -> None:
    config = _FakeRllibConfig()

    configure_rllib_evaluation(
        config,
        num_env_runners=2,
        parallel_to_training=True,
        env_config={"max_steps": 96},
        episodes=10,
        seed=142,
        num_envs_per_env_runner=4,
    )

    assert config.options["evaluation_interval"] == 1
    assert config.options["evaluation_parallel_to_training"] is True
    assert isinstance(
        config.options["custom_evaluation_function"],
        AnySearchEvaluationFunction,
    )


def test_rllib_parallel_evaluation_uses_native_scheduler() -> None:
    config = _FakeRllibConfig()

    configure_rllib_evaluation(
        config,
        num_env_runners=2,
        parallel_to_training=True,
        env_config={"max_steps": 96},
        episodes=10,
        seed=142,
        num_envs_per_env_runner=4,
    )

    assert config.options["evaluation_interval"] == 1
    assert config.options["evaluation_parallel_to_training"] is True
    assert isinstance(
        config.options["custom_evaluation_function"],
        AnySearchEvaluationFunction,
    )
