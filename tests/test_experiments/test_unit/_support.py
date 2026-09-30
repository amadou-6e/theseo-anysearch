"""Experiment test support objects.

Examples
--------
Patch the experiment runner to use a fake PPO algorithm.

>>> fake_build, orig = patch_build(runner_mod)
>>> callable(fake_build)
True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from theseo_anysearch.experiments.trajectory import VoxelEpisodeData


class FakeAlgo:
    """Duck-typed RLlib algorithm used by experiment runner tests."""

    def __init__(self) -> None:
        self._step = 0
        self.saved: list[str] = []
        self.restored: list[str] = []
        self.weights: dict[str, int] = {"step": 0}
        self.stopped = False

    def train(self) -> dict[str, Any]:
        """Return deterministic train results in RLlib's new API shape.

        Returns
        -------
        dict[str, Any]
            Synthetic metrics keyed like RLlib algorithm results.
        """

        self._step += 1
        self.weights = {"step": self._step}
        self._anysearch_evaluation_episodes = [
            VoxelEpisodeData(
                agent_count=1,
                max_steps=1,
                obs_mode="scalar",
                init_filled=[],
                steps=[],
                total_reward=0.0,
                success=False,
                start_pos=(1, 1, 1),
                goal_pos=(2, 2, 2),
            )
        ]
        return {
            "env_runners": {
                "episode_return_mean": float(self._step),
                "episode_len_mean": 20.0,
                "num_episodes_lifetime": self._step * 3,
                "num_env_steps_sampled_lifetime": self._step * 20,
            },
            "training_iteration": self._step,
        }

    def save(self, path: str) -> str:
        """Write a sentinel checkpoint for resume tests.

        Parameters
        ----------
        path : str
            Checkpoint directory path.

        Returns
        -------
        str
            The checkpoint path.
        """

        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.joinpath("weights.sentinel").write_text("ok", encoding="utf-8")
        self.saved.append(path)
        return path

    def restore(self, path: str) -> None:
        """Validate that the sentinel checkpoint exists.

        Parameters
        ----------
        path : str
            Checkpoint directory path.
        """

        sentinel = Path(path).joinpath("weights.sentinel")
        if not sentinel.exists():
            raise FileNotFoundError(path)
        self.restored.append(path)

    def get_weights(self) -> dict[str, int]:
        return dict(self.weights)

    def set_weights(self, weights: dict[str, int]) -> None:
        self.weights = dict(weights)

    def stop(self) -> None:
        self.stopped = True


def patch_build(runner_mod, fake_algo_cls=FakeAlgo):
    """Create a runner patch that swaps in a fake PPO-backed trainer.

    Parameters
    ----------
    runner_mod : module
        Imported ``theseo_anysearch.experiments.runner`` module.
    fake_algo_cls : type, default=FakeAlgo
        Fake algorithm class to instantiate from the fake PPO trainer.

    Returns
    -------
    tuple
        ``(fake_build, original_build)`` patch pair for ``_build_trainer``.
    """

    from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer

    class _FakePPO(PPOTrainer):
        """Minimal PPO trainer wrapper that returns a fake algorithm."""

        def _build_algorithm(self_inner):
            return fake_algo_cls()

    original = runner_mod._build_trainer

    def fake_build(cfg, output_dir):
        settings = cfg.to_settings()
        settings = settings.model_copy(
            update={"training": settings.training.model_copy(update={"output_dir": output_dir})}
        )
        return _FakePPO(settings)

    return fake_build, original
