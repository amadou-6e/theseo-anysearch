"""Deterministic trajectory collection on RLlib evaluation EnvRunners."""

from __future__ import annotations

from functools import partial
from typing import Any
import weakref


class AnySearchEvaluationFunction:
    """Bridge AnySearch deterministic episodes into RLlib evaluation scheduling."""

    def __init__(
        self,
        *,
        env_config: dict[str, Any],
        episodes: int,
        seed: int,
        multi_agent: bool,
        num_envs_per_env_runner: int,
    ) -> None:
        self._algorithm_ref: Any = None
        self._env_config = dict(env_config)
        self._episodes = episodes
        self._seed = seed
        self._multi_agent = multi_agent
        self._num_envs_per_env_runner = num_envs_per_env_runner

    def bind(self, algorithm: Any) -> None:
        """Bind the built Algorithm required by RLlib's legacy callback API."""
        self._algorithm_ref = weakref.ref(algorithm)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the runtime-only Algorithm reference from checkpoints."""
        state = dict(self.__dict__)
        state["_algorithm_ref"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore serializable callback configuration without runtime binding."""
        self.__dict__.update(state)

    def __call__(self, *args: Any) -> Any:
        """Collect episodes using either RLlib custom-evaluation signature."""
        algorithm = args[0] if args else (
            self._algorithm_ref() if self._algorithm_ref is not None else None
        )
        if algorithm is None:
            raise RuntimeError("AnySearch evaluation callback is not bound")
        episodes = collect_rllib_evaluation_episodes(
            algorithm,
            self._env_config,
            self._episodes,
            seed=self._seed,
            multi_agent=self._multi_agent,
            num_envs_per_env_runner=self._num_envs_per_env_runner,
            sync_weights=False,
        )
        algorithm._anysearch_evaluation_episodes = episodes
        env_steps = sum(len(episode.steps) for episode in episodes)
        rewards = [float(episode.total_reward) for episode in episodes]
        metrics = {
            "episode_reward_mean": sum(rewards) / len(rewards),
            "episode_len_mean": env_steps / len(episodes),
            "num_episodes": len(episodes),
        }
        if args:
            return metrics, env_steps, env_steps
        return metrics



class _PolicyAdapter:
    """Expose either an RLModule or legacy Policy for deterministic inference."""

    def __init__(self, env_runner: Any, observation_space: Any | None = None) -> None:
        self._env_runner = env_runner
        self._observation_space = observation_space

    def _rl_module(self) -> Any | None:
        """Return the new-stack inference module when one is available."""
        module = getattr(self._env_runner, "module", None)
        if module is None and hasattr(self._env_runner, "get_module"):
            module = self._env_runner.get_module()
        return module

    def _module_observations(self, observations: list[Any]) -> list[Any]:
        """Apply the same structured-observation flattening as Connector V2."""
        from gymnasium.spaces import flatten

        space = self._observation_space
        if space is None:
            runner = getattr(self._env_runner, "env_runner", self._env_runner)
            env = getattr(runner, "env", None)
            space = getattr(env, "single_observation_space", None)
        if space is None:
            return observations
        return [flatten(space, observation) for observation in observations]

    def _compute_module_actions(self, observations: list[Any]) -> Any:
        """Run a deterministic batched forward pass through an RLModule."""
        import torch
        import tree
        from ray.rllib.core.columns import Columns

        module = self._rl_module()
        if module is None:
            raise RuntimeError("RLModule inference requested without a module")
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        stacked = _stack_observations(self._module_observations(observations))
        tensor_obs = tree.map_structure(
            lambda value: torch.as_tensor(value, device=device),
            stacked,
        )
        with torch.inference_mode():
            outputs = module.forward_inference({Columns.OBS: tensor_obs})
            actions = outputs.get(Columns.ACTIONS)
            if actions is None:
                distribution_class = module.get_inference_action_dist_cls()
                distribution = distribution_class.from_logits(
                    outputs[Columns.ACTION_DIST_INPUTS]
                ).to_deterministic()
                actions = distribution.sample()
        return actions.detach().cpu().numpy()

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
        if self._rl_module() is not None:
            return self._compute_module_actions([observation])[0]

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
        if self._rl_module() is not None:
            return self._compute_module_actions(observations), [], {}

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
    sync_weights: bool = True,
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
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

        probe_env = VoxelEnv(env_config)
        observation_space = probe_env.observation_space
        probe_env.close()
        policy = _PolicyAdapter(
            algorithm,
            observation_space=observation_space,
        )
        if num_envs_per_env_runner > 1:
            episodes: list[Any] = []
            seeds = tuple(seed + episode_index for episode_index in range(count))
            for offset in range(0, count, num_envs_per_env_runner):
                episodes.extend(collect_vectorized_eval_episodes(
                    policy,
                    env_config,
                    seeds[offset:offset + num_envs_per_env_runner],
                ))
            return episodes
        return collect_eval_episodes(policy, env_config, count, seed=seed)

    active_worker_ids = worker_ids[: min(len(worker_ids), count)]
    assignments: list[list[int]] = [[] for _ in active_worker_ids]
    for episode_index in range(count):
        assignments[episode_index % len(assignments)].append(seed + episode_index)

    if sync_weights:
        sync_rllib_evaluation_weights(algorithm)
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


def sync_rllib_evaluation_weights(algorithm: Any) -> None:
    """Copy one stable policy snapshot to dedicated evaluation workers."""
    group = getattr(algorithm, "eval_env_runner_group", None)
    if group is None or not group.healthy_env_runner_ids():
        raise RuntimeError("parallel evaluation requires healthy evaluation workers")
    config = getattr(algorithm, "config", None)
    uses_learner = bool(
        getattr(config, "enable_rl_module_and_learner", False)
    )
    source = (
        getattr(algorithm, "learner_group", None)
        if uses_learner
        else getattr(algorithm, "env_runner", None)
    )
    group.sync_weights(
        from_worker_or_learner_group=source,
        inference_only=True,
    )


def configure_rllib_evaluation(
    rllib_config: Any,
    *,
    num_env_runners: int,
    parallel_to_training: bool = False,
    frequency: int = 1,
    env_config: dict[str, Any] | None = None,
    episodes: int = 1,
    seed: int = 42,
    multi_agent: bool = False,
    num_envs_per_env_runner: int = 1,
) -> Any:
    """Attach the dedicated RLlib evaluation EnvRunner pool to an algorithm config."""
    custom_function = AnySearchEvaluationFunction(
        env_config=env_config or {},
        episodes=episodes,
        seed=seed,
        multi_agent=multi_agent,
        num_envs_per_env_runner=num_envs_per_env_runner,
    )
    options = {
        "evaluation_interval": frequency,
        "evaluation_num_env_runners": num_env_runners,
        "evaluation_parallel_to_training": parallel_to_training,
        "evaluation_config": {"explore": False},
        "custom_evaluation_function": custom_function,
    }
    return rllib_config.evaluation(**options)


def bind_anysearch_evaluation_function(algorithm: Any) -> Any:
    """Bind a configured custom evaluator after RLlib builds the Algorithm."""
    function = getattr(algorithm.config, "custom_evaluation_function", None)
    if isinstance(function, AnySearchEvaluationFunction):
        function.bind(algorithm)
    return algorithm
