"""Policy scoring interfaces used by explainability backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from typing import Mapping, Sequence

import numpy as np

from theseo_anysearch.rllib.explain.features import action_directions_26
from theseo_anysearch.rllib.explain.models import ActionScoreTable


class PolicyScorer(ABC):
    """Base class for policy action scoring."""

    algorithm = "unknown"
    score_type = "score"

    @abstractmethod
    def score_all(self, observations: Sequence[Mapping[str, np.ndarray]]) -> ActionScoreTable:
        """Return scores for every action and observation."""

    def score_action(
        self,
        observations: Sequence[Mapping[str, np.ndarray]],
        actions: Sequence[int],
    ) -> np.ndarray:
        """Return selected action scores for each observation."""

        table = self.score_all(observations).values
        return np.asarray([table[row, action] for row, action in enumerate(actions)], dtype=np.float32)

    def select_action(self, observation: Mapping[str, np.ndarray]) -> int:
        """Return the greedy action for one observation.

        Parameters
        ----------
        observation : Mapping[str, np.ndarray]
            Observation dictionary.

        Returns
        -------
        int
            Highest-scoring action index.
        """

        scores = self.score_all([observation]).values[0]
        return int(np.argmax(scores))


class MockPolicyScorer(PolicyScorer):
    """Deterministic scorer backed by an explicit score table.

    Parameters
    ----------
    scores : np.ndarray
        One-dimensional 26-action score vector or a two-dimensional score table.
    score_type : str, default="mock_score"
        Score type label for reports.
    """

    algorithm = "mock"

    def __init__(self, scores: np.ndarray, score_type: str = "mock_score") -> None:
        self._scores = np.asarray(scores, dtype=np.float32)
        self.score_type = score_type
        if self._scores.ndim not in {1, 2}:
            raise ValueError("mock scores must be one- or two-dimensional")
        if self._scores.shape[-1] < 1:
            raise ValueError("mock scores must provide at least one action score")

    def score_all(self, observations: Sequence[Mapping[str, np.ndarray]]) -> ActionScoreTable:
        """Return explicit mock scores for all observations."""

        batch = len(observations)
        if self._scores.ndim == 1:
            values = np.repeat(self._scores.reshape(1, -1), batch, axis=0)
        elif self._scores.shape[0] == batch:
            values = self._scores
        elif self._scores.shape[0] == 1:
            values = np.repeat(self._scores, batch, axis=0)
        else:
            raise ValueError(f"mock score rows {self._scores.shape[0]} do not match batch {batch}")
        return ActionScoreTable(values=values.astype(np.float32, copy=False), score_type=self.score_type)


class LinearMockPolicyScorer(PolicyScorer):
    """Small heuristic scorer for backend smoke tests.

    The scorer rewards alignment with ``goal_direction`` and penalizes visible
    ray hits for each action.  It is intentionally simple and deterministic.

    Parameters
    ----------
    collision_penalty : float, default=2.0
        Penalty applied to ``ray_hits[action]``.
    type_penalty : float, default=0.5
        Penalty applied to ``ray_hit_types[action]``.
    """

    algorithm = "mock"
    score_type = "mock_linear_score"

    def __init__(self, collision_penalty: float = 2.0, type_penalty: float = 0.5) -> None:
        self._directions = np.asarray(action_directions_26(), dtype=np.float32)
        self._collision_penalty = collision_penalty
        self._type_penalty = type_penalty

    def score_all(self, observations: Sequence[Mapping[str, np.ndarray]]) -> ActionScoreTable:
        """Return heuristic scores for every radial action."""

        rows: list[np.ndarray] = []
        for observation in observations:
            goal_direction = np.asarray(observation["goal_direction"], dtype=np.float32)
            ray_hits = np.asarray(observation["ray_hits"], dtype=np.float32)
            ray_hit_types = np.asarray(observation["ray_hit_types"], dtype=np.float32)
            alignment = self._directions @ goal_direction
            scores = alignment - self._collision_penalty * ray_hits - self._type_penalty * ray_hit_types
            rows.append(scores.astype(np.float32, copy=False))
        return ActionScoreTable(values=np.stack(rows), score_type=self.score_type)


class DQNPolicyScorer(PolicyScorer):
    """Score observations with a restored RLlib DQN policy.

    Parameters
    ----------
    algorithm : Any
        Restored RLlib Algorithm-like object exposing ``compute_single_action``.
    policy_id : str, default="default_policy"
        RLlib policy id used for single-agent scoring.
    """

    algorithm = "dqn"
    score_type = "q_value"

    def __init__(
        self,
        algorithm: Any | None = None,
        policy_id: str = "default_policy",
        *,
        module: Any | None = None,
        observation_space: Any | None = None,
        action_count: int | None = None,
    ) -> None:
        if algorithm is None and module is None:
            raise ValueError("DQN scorer requires an Algorithm or RLModule")
        self._algorithm = algorithm
        self._policy_id = policy_id
        self._module = module
        self._observation_space = observation_space
        self._action_count = action_count

    @classmethod
    def from_module_checkpoint(
        cls,
        checkpoint_dir: Path,
        observation_space: Any,
        policy_id: str = "default_policy",
    ) -> "DQNPolicyScorer":
        """Restore only the DQN RLModule, avoiding a Ray algorithm runtime."""
        from ray.rllib.core.rl_module.rl_module import RLModule

        module_path = checkpoint_dir.joinpath(
            "learner_group", "learner", "rl_module", policy_id
        )
        if not module_path.joinpath("metadata.json").is_file():
            raise FileNotFoundError(f"RLModule checkpoint not found: {module_path}")
        module = RLModule.from_checkpoint(module_path)
        return cls(module=module, observation_space=observation_space, policy_id=policy_id)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        checkpoint: str = "latest",
        policy_id: str = "default_policy",
    ) -> "DQNPolicyScorer":
        """Restore a DQN scorer from an AnySearch run directory.

        Parameters
        ----------
        run_dir : Path
            Run directory containing ``experiment.yaml`` and ``checkpoints``.
        checkpoint : str, default="latest"
            Checkpoint selector: ``"latest"``, iteration number, checkpoint
            directory name, or explicit path.
        policy_id : str, default="default_policy"
            RLlib policy id.

        Returns
        -------
        DQNPolicyScorer
            Scorer backed by the restored RLlib algorithm.
        """

        from theseo_anysearch.experiments.loader import load_experiment
        from theseo_anysearch.experiments.models import ExperimentConfig

        experiment_path = run_dir.joinpath("experiment.yaml")
        experiment = load_experiment(experiment_path)
        if not isinstance(experiment, ExperimentConfig):
            raise ValueError(f"run directory {run_dir} does not contain a single experiment config")
        experiment = cls.resolve_run_geometry_pool(experiment, experiment_path)
        checkpoint_dir = cls.resolve_checkpoint_dir(run_dir, checkpoint)
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

        env_config = experiment.env.to_runtime_dict()
        curriculum = experiment.env.waypoint_curriculum
        if curriculum is not None and curriculum.enabled:
            env_config["waypoint_curriculum"] = curriculum.model_dump(mode="python")
        env = VoxelEnv(env_config)
        try:
            observation_space = env.observation_space
        finally:
            env.close()
        return cls.from_module_checkpoint(
            checkpoint_dir,
            observation_space,
            policy_id=policy_id,
        )

    @classmethod
    def from_checkpoint(
        cls,
        settings: Any,
        checkpoint_dir: Path,
        policy_id: str = "default_policy",
    ) -> "DQNPolicyScorer":
        """Restore a DQN scorer from settings and a checkpoint directory.

        Parameters
        ----------
        settings : Any
            Project ``Settings`` object used to rebuild the DQN algorithm.
        checkpoint_dir : Path
            Project checkpoint directory containing optional ``state.json``.
        policy_id : str, default="default_policy"
            RLlib policy id.

        Returns
        -------
        DQNPolicyScorer
            Scorer backed by the restored RLlib algorithm.
        """

        from theseo_anysearch.rllib.algorithms.dqn import DQNTrainer

        algorithm = DQNTrainer.build_algorithm_from_settings(settings)
        algorithm.restore(cls.resolve_rllib_checkpoint_path(checkpoint_dir))
        return cls(algorithm, policy_id=policy_id)

    @staticmethod
    def resolve_run_geometry_pool(experiment: Any, experiment_path: Path) -> Any:
        """Resolve geometry pool paths in copied run-local experiment YAMLs."""

        pool_cfg = getattr(experiment.env, "geometry_pool", None)
        if not isinstance(pool_cfg, dict) or not pool_cfg.get("pool_dir"):
            return experiment

        pool_dir = Path(str(pool_cfg["pool_dir"]))
        if pool_dir.is_absolute():
            return experiment

        candidates = DQNPolicyScorer.geometry_pool_candidates(pool_dir, experiment_path)
        for candidate in candidates:
            if candidate.joinpath("pool_meta.json").exists():
                new_pool = {**pool_cfg, "pool_dir": str(candidate)}
                return experiment.model_copy(
                    update={"env": experiment.env.model_copy(update={"geometry_pool": new_pool})}
                )
        return experiment

    @staticmethod
    def geometry_pool_candidates(pool_dir: Path, experiment_path: Path) -> list[Path]:
        """Return plausible absolute paths for a relative geometry pool."""

        candidates = [experiment_path.parent.joinpath(pool_dir).resolve()]
        repo_root = Path(__file__).resolve().parents[3]
        parts = pool_dir.parts
        if "runtime" in parts:
            runtime_index = parts.index("runtime")
            candidates.append(repo_root.joinpath(*parts[runtime_index:]).resolve())
        candidates.append(repo_root.joinpath(pool_dir).resolve())
        return list(dict.fromkeys(candidates))

    @staticmethod
    def resolve_checkpoint_dir(run_dir: Path, checkpoint: str) -> Path:
        """Resolve a project checkpoint selector to a checkpoint directory."""

        checkpoint_root = run_dir.joinpath("checkpoints")
        if checkpoint == "latest":
            latest_path = checkpoint_root.joinpath("latest.json")
            if not latest_path.exists():
                raise ValueError(f"latest checkpoint pointer not found: {latest_path}")
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            return Path(str(latest["path"]))

        candidate = Path(str(checkpoint))
        if candidate.exists():
            return candidate

        if checkpoint.isdigit():
            return checkpoint_root.joinpath(f"iter_{int(checkpoint):06d}")

        named = checkpoint_root.joinpath(checkpoint)
        if named.exists():
            return named

        raise ValueError(f"checkpoint {checkpoint!r} could not be resolved under {checkpoint_root}")

    @staticmethod
    def resolve_rllib_checkpoint_path(checkpoint_dir: Path) -> str:
        """Return the concrete RLlib restore path for a project checkpoint."""

        state_path = checkpoint_dir.joinpath("state.json")
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            return str(state.get("rllib_path", str(checkpoint_dir)))
        return str(checkpoint_dir)

    def score_all(self, observations: Sequence[Mapping[str, np.ndarray]]) -> ActionScoreTable:
        """Return DQN Q-values for every action and observation."""

        values = np.stack([self._q_values(observation) for observation in observations]).astype(
            np.float32,
            copy=False,
        )
        return ActionScoreTable(values=values, score_type=self.score_type)

    def _q_values(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compute and extract one action Q-value vector."""

        if self._module is not None:
            return self._module_q_values(observation)

        result = self._compute_action(observation)
        info = self._extract_info(result)
        q_values = self._extract_q_values(info)
        expected = self.action_count
        if q_values.shape != (expected,):
            raise ValueError(
                f"DQN scorer expected {expected} Q-values, got shape {q_values.shape}"
            )
        return q_values

    @property
    def action_count(self) -> int:
        """Return the action count exposed by the restored policy."""
        source = self._module if self._module is not None else self._algorithm
        action_space = getattr(source, "action_space", None)
        if (
            action_space is None
            and self._algorithm is not None
            and hasattr(self._algorithm, "get_policy")
        ):
            action_space = self._algorithm.get_policy(self._policy_id).action_space
        count = getattr(action_space, "n", None)
        if count is not None:
            return int(count)
        if self._action_count is not None:
            return self._action_count
        raise ValueError(
            "DQN scorer could not determine a discrete action space from the "
            "restored algorithm/module; pass action_count explicitly"
        )

    def select_action(self, observation: Mapping[str, np.ndarray]) -> int:
        """Return RLlib's deterministic DQN action for one observation."""

        result = self._compute_action(observation)
        if isinstance(result, tuple):
            return int(result[0])
        if isinstance(result, np.ndarray):
            return int(result.item())
        return int(result)

    def _compute_action(self, observation: Mapping[str, np.ndarray]) -> Any:
        """Call RLlib action inference with deterministic full fetch enabled."""

        if self._module is not None:
            return int(np.argmax(self._module_q_values(observation)))

        try:
            return self._algorithm.compute_single_action(
                observation,
                policy_id=self._policy_id,
                explore=False,
                full_fetch=True,
            )
        except TypeError:
            return self._algorithm.compute_single_action(
                observation,
                explore=False,
                full_fetch=True,
            )

    def _extract_info(self, result: Any) -> Mapping[str, Any]:
        """Extract the info dictionary returned by ``compute_single_action``."""

        if isinstance(result, tuple) and len(result) >= 3 and isinstance(result[2], Mapping):
            return result[2]
        if isinstance(result, Mapping):
            return result
        raise ValueError(
            "DQN scorer requires compute_single_action(..., full_fetch=True) to return policy info"
        )

    def _extract_q_values(self, info: Mapping[str, Any]) -> np.ndarray:
        """Extract Q-values from RLlib policy info."""

        candidates = (
            info.get("q_values"),
            info.get("action_dist_inputs"),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            values = np.asarray(candidate, dtype=np.float32).reshape(-1)
            if values.size == self.action_count:
                return values
        keys = sorted(str(key) for key in info.keys())
        raise ValueError(
            f"DQN policy info did not contain {self.action_count} Q-values; "
            f"available keys: {keys}"
        )

    def _module_q_values(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compute Q-values directly through a restored modern RLModule."""
        if self._observation_space is None:
            raise ValueError("RLModule scoring requires the structured observation space")
        import torch
        from gymnasium.spaces import flatten
        from ray.rllib.core.columns import Columns
        from ray.rllib.algorithms.dqn.default_dqn_rl_module import QF_PREDS

        flat = flatten(self._observation_space, observation)
        device = next(self._module.parameters()).device
        batch = {Columns.OBS: torch.as_tensor(flat, device=device).unsqueeze(0)}
        self._module.eval()
        with torch.inference_mode():
            output = self._module.compute_q_values(batch)
        return output[QF_PREDS][0].detach().cpu().numpy().astype(np.float32)
