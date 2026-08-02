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
        """Compute one action through the adapted RLlib policy.

        Parameters
        ----------
        observation : Any
            Environment observation.
        policy_id : str
            RLlib policy identifier.
        explore : bool
            Whether policy inference may explore.

        Returns
        -------
        Any
            Policy action.
        """
        policy = self._env_runner.get_policy(policy_id)
        observation = self._preprocess_observation(
            policy, observation, env_id="0"
        )
        return policy.compute_single_action(obs=observation, explore=explore)

    def compute_actions(
        self,
        observations: list[Any],
        *,
        policy_id: str = "default_policy",
        explore: bool = False,
    ) -> Any:
        """Compute a batch of actions through the adapted RLlib policy.

        Parameters
        ----------
        observations : list[Any]
            Environment observations to batch.
        policy_id : str
            RLlib policy identifier.
        explore : bool
            Whether policy inference may explore.

        Returns
        -------
        Any
            Batched policy actions.
        """
        policy = self._env_runner.get_policy(policy_id)
        processed = [
            self._preprocess_observation(policy, observation, env_id=str(index))
            for index, observation in enumerate(observations)
        ]
        return policy.compute_actions(
            _stack_observations(processed),
            explore=explore,
        )

    @staticmethod
    def _preprocess_observation(
        policy: Any,
        observation: Any,
        *,
        env_id: str,
    ) -> Any:
        from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
        from ray.rllib.utils.typing import AgentConnectorDataType

        preprocessors = policy.agent_connectors[ObsPreprocessorConnector]
        if preprocessors:
            if len(preprocessors) != 1:
                raise RuntimeError("expected one RLlib observation preprocessor")
            preprocessor = preprocessors[0]
            if not preprocessor.is_identity():
                preprocessor.in_eval()
                preprocessor.reset(env_id=env_id)
                connector_data = AgentConnectorDataType(
                    env_id,
                    "0",
                    {"obs": observation},
                )
                observation = preprocessor([connector_data])[0].data["obs"]
        return observation


def _stack_observations(observations: list[Any]) -> Any:
    """Stack nested observations into the batch shape expected by RLlib."""
    import numpy as np

    first = observations[0]
    if isinstance(first, dict):
        return {
            key: _stack_observations([observation[key] for observation in observations])
            for key in first
        }
    if isinstance(first, tuple):
        return tuple(
            _stack_observations([observation[index] for observation in observations])
            for index in range(len(first))
        )
    return np.stack(observations)


def _collect_worker_episodes(
    env_runner: Any,
    *,
    seeds: tuple[int, ...],
    env_config: dict[str, Any],
    multi_agent: bool,
    num_envs_per_env_runner: int,
) -> list[tuple[int, Any]]:
    """Collect assigned seeds inside one RLlib evaluation actor."""
    from theseo_anysearch.experiments.trajectory import (
        collect_eval_episode,
        collect_multi_eval_episode,
        collect_vectorized_eval_episodes,
    )

    collector = collect_multi_eval_episode if multi_agent else collect_eval_episode
    policy = _PolicyAdapter(env_runner)
    if not multi_agent and num_envs_per_env_runner > 1:
        results: list[tuple[int, Any]] = []
        for offset in range(0, len(seeds), num_envs_per_env_runner):
            batch_seeds = seeds[offset:offset + num_envs_per_env_runner]
            episodes = collect_vectorized_eval_episodes(
                policy, env_config, batch_seeds
            )
            results.extend(zip(batch_seeds, episodes))
        return results
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
    num_envs_per_env_runner: int = 1,
) -> list[Any]:
    """Collect a deterministic batch concurrently on RLlib evaluation workers."""
    if count < 1:
        raise ValueError("evaluation episode count must be at least one")
    if num_envs_per_env_runner < 1:
        raise ValueError("evaluation environments per runner must be at least one")

    group = getattr(algorithm, "eval_env_runner_group", None)
    worker_ids = group.healthy_env_runner_ids() if group is not None else []
    if not worker_ids:
        from theseo_anysearch.experiments.trajectory import (
            collect_eval_episodes,
            collect_multi_eval_episode,
            collect_vectorized_eval_episodes,
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
        if num_envs_per_env_runner > 1:
            episodes: list[Any] = []
            seeds = tuple(seed + episode_index for episode_index in range(count))
            policy = _PolicyAdapter(algorithm)
            for offset in range(0, count, num_envs_per_env_runner):
                episodes.extend(collect_vectorized_eval_episodes(
                    policy,
                    env_config,
                    seeds[offset:offset + num_envs_per_env_runner],
                ))
            return episodes
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
            num_envs_per_env_runner=num_envs_per_env_runner,
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
