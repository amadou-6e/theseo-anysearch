"""Shared trainer logic for building, training, checkpointing, and trajectory recording."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.models import Settings


def _log_trainer_stage(message: str) -> None:
    """Print a timestamped trainer progress message for foreground debugging."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[trainer] {ts} {message}", flush=True)


def _append_trainer_stage_log(output_dir: Path, message: str) -> None:
    """Append a timestamped trainer stage marker to the run-local debug log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = output_dir.joinpath("debug_stage.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[trainer] {ts} {message}\n")


def _resolve_pool_dir(geometry_pool: dict | None) -> dict | None:
    """Return geometry_pool with pool_dir resolved to an absolute path.

    The CLI resolves relative pool_dir values before calling any trainer, so
    this is typically a no-op.  It exists as a safety net for callers that
    build trainers directly (e.g. tests) with relative paths.
    """
    if not geometry_pool or not geometry_pool.get("pool_dir"):
        return geometry_pool
    pool_dir = Path(str(geometry_pool["pool_dir"]))
    if not pool_dir.is_absolute():
        pool_dir = Path.cwd() / pool_dir
    return {**geometry_pool, "pool_dir": str(pool_dir.resolve())}


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


def _detect_num_gpus(require_gpu: bool = False, *, num_gpus: float | None = None) -> float:
    """Return GPU count to allocate per RLlib algorithm.

    If *num_gpus* is provided it is returned directly (allows fractional values
    for concurrent Tune trials sharing one GPU, e.g. 0.5).
    When *require_gpu* is True, raises AssertionError if no CUDA device is found.
    Install hint: pip install .[torch-gpu] --index-url https://download.pytorch.org/whl/cu124
    """
    if num_gpus is not None:
        return num_gpus
    import torch

    n = torch.cuda.device_count()
    if require_gpu:
        assert n > 0, (
            f"require_gpu=True but torch.cuda.device_count()={n}. "
            "Install CUDA-enabled torch: pip install .[torch-gpu] "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
    return n


class _TensorBoardRunWriter:
    """Write per-iteration training metrics to a run-local TensorBoard logdir.

    The writer keeps TensorBoard ownership in the project trainer layer instead of
    relying on RLlib's legacy logger placement, which may write to opaque Ray temp
    locations. Event files are emitted under ``<run_dir>/tensorboard`` so
    ``anysearch tensorboard`` can point at the run directory directly.
    """

    def __init__(self, output_dir: Path) -> None:
        self._log_dir = output_dir.joinpath("tensorboard")
        self._writer: Any | None = None

        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            return

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

    @property
    def enabled(self) -> bool:
        """Return True when TensorBoard writing is active."""
        return self._writer is not None

    @property
    def log_dir(self) -> Path:
        """Return the concrete TensorBoard log directory."""
        return self._log_dir

    def log_iteration(self, result: "TrainResult") -> None:
        """Append one iteration of scalar metrics to the event stream."""
        if self._writer is None:
            return

        self._writer.add_scalar(
            "train/episode_reward_mean",
            result.episode_reward_mean,
            result.iteration,
        )
        self._writer.add_scalar(
            "train/episode_len_mean",
            result.episode_len_mean,
            result.iteration,
        )
        self._writer.add_scalar(
            "train/episodes_total",
            result.episodes_total,
            result.iteration,
        )
        self._writer.add_scalar(
            "train/elapsed_s",
            result.elapsed_s,
            result.iteration,
        )
        self._writer.flush()

    def log_scalars(self, iteration: int, scalars: Mapping[str, float]) -> None:
        """Append additional scalar metrics for one iteration."""
        if self._writer is None:
            return

        for tag, value in scalars.items():
            self._writer.add_scalar(tag, value, iteration)
        self._writer.flush()

    def close(self) -> None:
        """Close the underlying SummaryWriter when it exists."""
        if self._writer is None:
            return
        self._writer.close()


class RllibTrainResult(BaseModel):
    """Normalized view over RLlib training results.

    Parameters
    ----------
    env_runners : dict[str, Any]
        New API stack result block for env runner metrics.
    episode_reward_mean : float | None
        Legacy top-level mean episode reward.
    episode_len_mean : float | None
        Legacy top-level mean episode length.
    episodes_total : int | None
        Legacy total number of episodes seen.
    training_iteration : int | None
        RLlib-reported training iteration.
    time_this_iter_s : float | None
        Duration of the iteration in seconds.
    """
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

    def _env_config_dict(self) -> dict:
        """Build the env config dict passed to VoxelEnv / MultiVoxelEnv."""
        env = self._config.env
        return {
            "stl_path": str(env.stl_path) if env.stl_path else None,
            "scale": env.scale,
            "scale_range": env.scale_range,
            "agent_count": env.agent_count,
            "max_steps": env.max_steps,
            "seed": env.seed,
            "obs_mode": env.obs_mode,
            "box_radius": env.box_radius,
            "box_radii": env.box_radii,
            "ray_max_len": env.ray_max_len,
            "grid_size": env.grid_size,
            "trail_mode": env.trail_mode,
            "geometry_boxes": env.geometry_boxes,
            "geometry_pool": _resolve_pool_dir(env.geometry_pool),
            "geometry_padding": env.geometry_padding,
            "waypoints_file": env.waypoints_file,
            "step_cost": env.step_cost,
            "collision_cost": env.collision_cost,
            "goal_reward": env.goal_reward,
            "distance_shaping": env.distance_shaping,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> list[TrainResult]:
        """
        Run the training loop for config.training.iterations steps.
        Returns a list of TrainResult, one per iteration.
        """
        import warnings

        if self._algo is None:
            _log_trainer_stage("Building algorithm instance")
            _append_trainer_stage_log(self._output_dir, "Building algorithm instance")
            self._algo = self._build_algorithm()
            _log_trainer_stage("Algorithm instance ready")
            _append_trainer_stage_log(self._output_dir, "Algorithm instance ready")

        training = self._config.training
        results: list[TrainResult] = []
        tb_writer = _TensorBoardRunWriter(self._output_dir)

        # --- Trajectory writer setup (lazy import to keep CLI fast) ---
        traj_every = training.trajectory_every
        best_traj = training.best_trajectory
        _traj_writer = None
        _is_multi = training.algorithm == "multi_agent_voxel_ppo"
        _env_cfg: dict | None = None
        if traj_every or best_traj:
            from theseo_anysearch.experiments.output import OutputStore
            from theseo_anysearch.experiments.trajectory import (
                EpisodeRunMetrics,
                MultiTrajectoryWriter,
                TrajectoryWriter,
            )
            _store = OutputStore(self._output_dir)
            _Writer = MultiTrajectoryWriter if _is_multi else TrajectoryWriter
            _traj_writer = _Writer(_store, traj_every, best_traj)
            _metrics_cls = EpisodeRunMetrics
            _run_id = self._output_dir.name
            _exp_name = self._output_dir.parent.name
            _env_cfg = self._env_config_dict()

        try:
            for i in range(self._iteration, training.iterations):
                _log_trainer_stage(
                    f"Starting train iteration {i + 1}/{training.iterations}"
                )
                _append_trainer_stage_log(
                    self._output_dir,
                    f"Starting train iteration {i + 1}/{training.iterations}",
                )
                t0 = time.perf_counter()
                rllib_result = self._algo.train()
                elapsed = time.perf_counter() - t0
                _log_trainer_stage(
                    f"Finished train iteration {i + 1}/{training.iterations} in {elapsed:.2f}s"
                )
                _append_trainer_stage_log(
                    self._output_dir,
                    f"Finished train iteration {i + 1}/{training.iterations} in {elapsed:.2f}s",
                )

                self._iteration = i + 1
                parsed = RllibTrainResult.from_raw(rllib_result)
                self._episodes_total = parsed.parse_episodes_total(
                ) or self._episodes_total

                result = TrainResult.from_rllib(self._iteration, rllib_result,
                                                elapsed)
                results.append(result)
                tb_writer.log_iteration(result)
                self.on_iteration_end(result)

                if self._iteration % training.checkpoint_interval == 0:
                    self.checkpoint()

                # --- Trajectory recording ---
                _is_last_iter = self._iteration == training.iterations
                if _traj_writer is not None and _env_cfg is not None:
                    try:
                        _log_trainer_stage(
                            f"Collecting eval trajectory for iteration {self._iteration}"
                        )
                        _append_trainer_stage_log(
                            self._output_dir,
                            f"Collecting eval trajectory for iteration {self._iteration}",
                        )
                        if _is_multi:
                            from theseo_anysearch.experiments.trajectory import collect_multi_eval_episode
                            episode = collect_multi_eval_episode(
                                self._algo, _env_cfg, seed=self._iteration
                            )
                        else:
                            from theseo_anysearch.experiments.trajectory import collect_eval_episode
                            episode = collect_eval_episode(
                                self._algo, _env_cfg, seed=self._iteration
                            )
                        if _is_multi:
                            metrics = _metrics_cls.from_multi_voxel_episode(episode)
                        else:
                            metrics = _metrics_cls.from_voxel_episode(episode)
                        tb_writer.log_scalars(self._iteration, metrics.as_scalar_dict())
                        _traj_writer.record(episode)
                        _traj_writer.on_iteration_end(
                            self._iteration,
                            result.episode_reward_mean,
                            _exp_name,
                            _run_id,
                            force=_is_last_iter,
                        )
                        _log_trainer_stage(
                            f"Recorded eval trajectory for iteration {self._iteration}"
                        )
                        _append_trainer_stage_log(
                            self._output_dir,
                            f"Recorded eval trajectory for iteration {self._iteration}",
                        )
                    except Exception as exc:
                        warnings.warn(
                            f"trajectory collection failed at iter {self._iteration}: {exc}",
                            stacklevel=2,
                        )
        finally:
            tb_writer.close()

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
