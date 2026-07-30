"""Deterministic trajectory collection on RLlib evaluation EnvRunners."""

from __future__ import annotations

from functools import partial
from typing import Any


class _PolicyAdapter:
    """Expose a RolloutWorker policy through the Algorithm inference interface."""

    def __init__(self, env_runner: Any) -> None:
        self._env_runner = env_runner

    def compute_single_action(
        self,
        observation: Any,
        *,
        policy_id: str = "default_policy",
        explore: bool = False,
    ) -> Any:
        from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
        from ray.rllib.utils.typing import AgentConnectorDataType

        policy = self._env_runner.get_policy(policy_id)
        preprocessors = policy.agent_connectors[ObsPreprocessorConnector]
        if preprocessors:
            if len(preprocessors) != 1:
                raise RuntimeError("expected one RLlib observation preprocessor")
            preprocessor = preprocessors[0]
            if not preprocessor.is_identity():
                preprocessor.in_eval()
                preprocessor.reset(env_id="0")
                connector_data = AgentConnectorDataType(
                    "0",
                    "0",
                    {"obs": observation},
                )
                observation = preprocessor([connector_data])[0].data["obs"]
        return policy.compute_single_action(obs=observation, explore=explore)


def _collect_worker_episodes(
    env_runner: Any,
    *,
    seeds: tuple[int, ...],
    env_config: dict[str, Any],
    multi_agent: bool,
) -> list[tuple[int, Any]]:
    """Collect assigned seeds inside one RLlib evaluation actor."""
    from theseo_anysearch.experiments.trajectory import (
        collect_eval_episode,
        collect_multi_eval_episode,
    )

    collector = collect_multi_eval_episode if multi_agent else collect_eval_episode
    policy = _PolicyAdapter(env_runner)
    return [
        (seed, collector(policy, env_config, seed=seed))
        for seed in seeds
    ]


def collect_rllib_evaluation_episodes(
    algorithm: Any,
    env_config: dict[str, Any],
    count: int,
    *,
    seed: int,
    multi_agent: bool,
) -> list[Any]:
    """Collect a deterministic batch concurrently on RLlib evaluation workers."""
    if count < 1:
        raise ValueError("evaluation episode count must be at least one")

    group = getattr(algorithm, "eval_env_runner_group", None)
    worker_ids = group.healthy_env_runner_ids() if group is not None else []
    if not worker_ids:
        from theseo_anysearch.experiments.trajectory import (
            collect_eval_episodes,
            collect_multi_eval_episode,
        )

        if multi_agent:
            return [
                collect_multi_eval_episode(
                    algorithm,
                    env_config,
                    seed=seed + episode_index,
                )
                for episode_index in range(count)
            ]
        return collect_eval_episodes(algorithm, env_config, count, seed=seed)

    active_worker_ids = worker_ids[: min(len(worker_ids), count)]
    assignments: list[list[int]] = [[] for _ in active_worker_ids]
    for episode_index in range(count):
        assignments[episode_index % len(assignments)].append(seed + episode_index)

    weights_source = getattr(algorithm, "env_runner", None)
    group.sync_weights(
        from_worker_or_learner_group=weights_source,
        inference_only=True,
    )
    functions = [
        partial(
            _collect_worker_episodes,
            seeds=tuple(worker_seeds),
            env_config=env_config,
            multi_agent=multi_agent,
        )
        for worker_seeds in assignments
    ]
    worker_results = group.foreach_env_runner(
        functions,
        local_env_runner=False,
        remote_worker_ids=active_worker_ids,
    )
    seeded_episodes = [
        seeded_episode
        for worker_result in worker_results
        for seeded_episode in worker_result
    ]
    if len(seeded_episodes) != count:
        raise RuntimeError(
            "RLlib evaluation workers returned "
            f"{len(seeded_episodes)} episodes; expected {count}"
        )
    seeded_episodes.sort(key=lambda item: item[0])
    return [episode for _, episode in seeded_episodes]


def configure_rllib_evaluation(rllib_config: Any, *, num_env_runners: int) -> Any:
    """Attach the dedicated RLlib evaluation EnvRunner pool to an algorithm config."""
    return rllib_config.evaluation(
        evaluation_interval=None,
        evaluation_num_env_runners=num_env_runners,
        evaluation_parallel_to_training=False,
        evaluation_config={"explore": False},
    )
