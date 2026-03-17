from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import ClassVar
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.models import Settings


def _patch_rllib_replay_buffer_type_check() -> None:
    """
    Work around a Ray bug in Algorithm._create_local_replay_buffer_if_necessary.

    After AlgorithmConfig.validate() resolves replay_buffer_config["type"] from a
    string to the actual class, the legacy check
        if "EpisodeReplayBuffer" in config["replay_buffer_config"]["type"]
    fails with TypeError because "in" on a class object is not supported.

    This patch converts the resolved class back to its name string before the
    check so that the string membership test works correctly.
    """
    try:
        import ray.rllib.algorithms.algorithm as _alg_mod
        _orig = _alg_mod.Algorithm._create_local_replay_buffer_if_necessary

        def _safe(self, config):  # type: ignore[override]
            rb = config.get("replay_buffer_config")
            if rb and not isinstance(rb.get("type"), str):
                config = dict(config, replay_buffer_config=dict(rb, type=rb["type"].__name__))
            return _orig(self, config)

        _alg_mod.Algorithm._create_local_replay_buffer_if_necessary = _safe
    except Exception:
        pass  # Ray not installed or API changed — skip silently


_patch_rllib_replay_buffer_type_check()


def _detect_num_gpus(require_gpu: bool = False) -> int:
    """Return GPU count available to Ray (0 or 1 for single-node).

    When *require_gpu* is True, raises AssertionError if no CUDA device is found.
    Install hint: pip install .[torch-gpu] --index-url https://download.pytorch.org/whl/cu124
    """
    import torch

    n = torch.cuda.device_count()
    if require_gpu:
        assert n > 0, (
            f"require_gpu=True but torch.cuda.device_count()={n}. "
            "Install CUDA-enabled torch: pip install .[torch-gpu] "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
    return n


class RllibTrainResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    env_runners: dict[str, Any] = Field(default_factory=dict)
    episode_reward_mean: float | None = None
    episode_len_mean: float | None = None
    episodes_total: int | None = None
    training_iteration: int | None = None
    time_this_iter_s: float | None = None

    @classmethod
    def from_raw(cls, result: dict[str, Any]) -> "RllibTrainResult":
        return cls.model_validate(result)

    # TODO! make these properties e.g. @property def mean_reward(self) → float
    def parse_episode_return(self) -> float:
        value = self.env_runners.get("episode_return_mean")
        if value is not None:
            return float(value)
        return float(self.episode_reward_mean
                     ) if self.episode_reward_mean is not None else 0.0

    def parse_episode_len(self) -> float:
        value = self.env_runners.get("episode_len_mean")
        if value is not None:
            return float(value)
        return float(self.episode_len_mean
                     ) if self.episode_len_mean is not None else 0.0

    def parse_episodes_total(self) -> int:
        value = self.env_runners.get("num_episodes_lifetime")
        if value is not None:
            return int(value)
        return int(
            self.episodes_total) if self.episodes_total is not None else 0


class TrainResult(BaseModel):
    """Project-level summary of one RLlib training iteration."""
    model_config = ConfigDict(extra="forbid")

    iteration: int
    episode_reward_mean: float
    episode_len_mean: float
    episodes_total: int
    elapsed_s: float
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_rllib(
        cls,
        iteration: int,
        rllib_result: dict[str, Any],
        elapsed_s: float,
    ) -> "TrainResult":
        """Build from the dict returned by ray.rllib.algorithms.Algorithm.train()."""
        parsed = RllibTrainResult.from_raw(rllib_result)
        return cls(
            iteration=iteration,
            episode_reward_mean=parsed.parse_episode_return(),
            episode_len_mean=parsed.parse_episode_len(),
            episodes_total=parsed.parse_episodes_total(),
            elapsed_s=elapsed_s,
            extra={
                "training_iteration": parsed.training_iteration or iteration,
                "time_this_iter_s": parsed.time_this_iter_s,
            },
        )


class Trainer(ABC):
    """
    Project-level wrapper around a Ray RLlib Algorithm.

    Subclasses implement _build_algorithm() to construct and return a
    configured ray.rllib.algorithms.Algorithm instance.  This base class
    handles the iteration loop, checkpointing, restore, and resume on top
    of whatever RLlib trainer is provided.

    For unit tests, _build_algorithm() can return any object that exposes
    .train() → dict, .save(path: str) → Any, .restore(path: str) → None.
    """

    algorithm_name: ClassVar[str | None] = None
    _registry: ClassVar[dict[str, type["Trainer"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.algorithm_name:
            cls._registry[cls.algorithm_name] = cls

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._algo: Any = None
        self._iteration: int = 0
        self._episodes_total: int = 0
        self._output_dir: Path = Path(config.training.output_dir)

    @classmethod
    def from_settings(cls, config: Settings) -> "Trainer":
        if cls is not Trainer:
            return cls(config)

        trainer_cls = cls._registry.get(config.training.algorithm.lower())
        if trainer_cls is None:
            raise NotImplementedError(
                f"No Trainer registered for algorithm '{config.training.algorithm}'."
            )
        return trainer_cls(config)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_algorithm(self) -> Any:
        """
        Construct the RLlib Algorithm (or compatible duck-typed object).
        Called once at the start of train() or restore().
        """

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    def on_iteration_end(self, result: TrainResult) -> None:
        """Called after every training iteration. Override to add callbacks."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> list[TrainResult]:
        """
        Run the training loop for config.training.iterations steps.
        Returns a list of TrainResult, one per iteration.
        """
        if self._algo is None:
            self._algo = self._build_algorithm()

        training = self._config.training
        results: list[TrainResult] = []

        for i in range(self._iteration, training.iterations):
            t0 = time.perf_counter()
            rllib_result = self._algo.train()
            elapsed = time.perf_counter() - t0

            self._iteration = i + 1
            parsed = RllibTrainResult.from_raw(rllib_result)
            self._episodes_total = parsed.parse_episodes_total(
            ) or self._episodes_total

            result = TrainResult.from_rllib(self._iteration, rllib_result,
                                            elapsed)
            results.append(result)
            self.on_iteration_end(result)

            if self._iteration % training.checkpoint_interval == 0:
                self.checkpoint()

        return results

    def checkpoint(self) -> Path:
        """
        Save a project checkpoint for the current iteration.
        Calls algo.save() for RLlib state, then writes state.json + latest.json.
        Returns the checkpoint directory path.
        """
        ckpt_dir = self._output_dir / "checkpoints" / f"iter_{self._iteration:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # RLlib legacy save() may return a subdirectory path (e.g. ckpt_dir/checkpoint_000001).
        # Store the returned path so restore() uses the exact location.
        rllib_path = str(ckpt_dir)
        if self._algo is not None:
            returned = self._algo.save(str(ckpt_dir))
            if isinstance(returned, str) and returned:
                rllib_path = returned

        self._write_state(ckpt_dir, rllib_path=rllib_path)
        self._write_latest_pointer(ckpt_dir)
        return ckpt_dir

    def restore(self, checkpoint_dir: Path) -> None:
        """Restore algorithm weights and project state from a checkpoint directory."""
        if self._algo is None:
            self._algo = self._build_algorithm()

        state_file = checkpoint_dir / "state.json"
        rllib_path = str(checkpoint_dir)
        if state_file.exists():
            state = json.loads(state_file.read_text())
            self._iteration = state["iteration"]
            self._episodes_total = state.get("episodes_total", 0)
            rllib_path = state.get("rllib_path", str(checkpoint_dir))

        self._algo.restore(rllib_path)

    def resume(self) -> bool:
        """
        Restore from the latest checkpoint if one exists.
        Returns True when a checkpoint was found and loaded, False otherwise.
        """
        latest_ptr = self._output_dir / "checkpoints" / "latest.json"
        if not latest_ptr.exists():
            return False
        info = json.loads(latest_ptr.read_text())
        self.restore(Path(info["path"]))
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_state(self,
                     checkpoint_dir: Path,
                     rllib_path: str | None = None) -> None:
        state = {
            "iteration": self._iteration,
            "episodes_total": self._episodes_total,
            "rllib_path": rllib_path or str(checkpoint_dir),
        }
        (checkpoint_dir / "state.json").write_text(json.dumps(state, indent=2))

    def _write_latest_pointer(self, checkpoint_dir: Path) -> None:
        ptr = {"path": str(checkpoint_dir), "iteration": self._iteration}
        (self._output_dir / "checkpoints" / "latest.json").write_text(
            json.dumps(ptr, indent=2))
