"""
Trainer tests — cover execution, config validity, output sanity, restore, and resume.

All tests use FakeAlgorithm (from conftest) so no Ray installation is required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unittest.mock import patch

from theseo_anysearch.rllib.trainer.results import IterationTimings, TrainResult
from theseo_anysearch.rllib.trainer.runtime import _detect_num_gpus
from theseo_anysearch.rllib.trainer.trainer import Trainer
from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer, _set_rllib_storage_path
from theseo_anysearch.experiments.trajectory import VoxelEpisodeData, VoxelStepData


# ---------------------------------------------------------------------------
# 0. _detect_num_gpus — GPU detection helper
# ---------------------------------------------------------------------------

class TestDetectNumGpus:
    """Tests DetectNumGpus."""
    def test_returns_zero_when_no_cuda(self):
        with patch("torch.cuda.device_count", return_value=0):
            assert _detect_num_gpus(require_gpu=False) == 0

    def test_returns_count_when_gpu_present(self):
        with patch("torch.cuda.device_count", return_value=1):
            assert _detect_num_gpus(require_gpu=False) == 1

    def test_require_gpu_false_does_not_raise_when_no_gpu(self):
        with patch("torch.cuda.device_count", return_value=0):
            _detect_num_gpus(require_gpu=False)  # must not raise

    def test_require_gpu_true_raises_when_no_gpu(self):
        with patch("torch.cuda.device_count", return_value=0):
            with pytest.raises(AssertionError, match="require_gpu"):
                _detect_num_gpus(require_gpu=True)

    def test_require_gpu_true_passes_when_gpu_present(self):
        with patch("torch.cuda.device_count", return_value=1):
            result = _detect_num_gpus(require_gpu=True)
        assert result == 1

    def test_assertion_message_contains_install_hint(self):
        with patch("torch.cuda.device_count", return_value=0):
            with pytest.raises(AssertionError, match="torch-gpu"):
                _detect_num_gpus(require_gpu=True)


def test_rllib_storage_path_uses_run_directory(tmp_path: Path):
    from ray.tune.trainable import trainable as ray_trainable

    storage_path = _set_rllib_storage_path(str(tmp_path))

    assert storage_path == Path(tmp_path, "rllib")
    assert ray_trainable.DEFAULT_STORAGE_PATH == str(storage_path)


# ---------------------------------------------------------------------------
# 0. TrainResult.from_rllib() — new and legacy API stack parsing
# ---------------------------------------------------------------------------

class TestTrainResult:
    """Tests TrainResult."""
    def test_new_api_stack_reads_env_runners(self):
        result = TrainResult.from_rllib(1, {"env_runners": {"episode_return_mean": 5.0, "episode_len_mean": 20.0, "num_episodes_lifetime": 10}}, 0.1)
        assert result.episode_reward_mean == pytest.approx(5.0)

    def test_legacy_stack_reads_top_level(self):
        result = TrainResult.from_rllib(1, {"episode_reward_mean": 3.0, "episode_len_mean": 15.0, "episodes_total": 6}, 0.1)
        assert result.episode_reward_mean == pytest.approx(3.0)

    def test_new_api_stack_takes_priority_over_legacy(self):
        result = TrainResult.from_rllib(1, {
            "env_runners": {"episode_return_mean": 7.0},
            "episode_reward_mean": 1.0,
        }, 0.1)
        assert result.episode_reward_mean == pytest.approx(7.0)

    def test_missing_reward_key_returns_zero(self):
        result = TrainResult.from_rllib(1, {}, 0.1)
        assert result.episode_reward_mean == pytest.approx(0.0)

    def test_iteration_field_set(self):
        result = TrainResult.from_rllib(4, {"episode_reward_mean": 1.0}, 0.5)
        assert result.iteration == 4

    def test_elapsed_s_set(self):
        result = TrainResult.from_rllib(1, {}, 1.234)
        assert result.elapsed_s == pytest.approx(1.234)

    def test_extra_contains_training_iteration(self):
        result = TrainResult.from_rllib(2, {"training_iteration": 2, "time_this_iter_s": 0.5}, 0.5)
        assert result.extra["training_iteration"] == 2

    def test_episodes_total_from_new_stack(self):
        result = TrainResult.from_rllib(1, {"env_runners": {"num_episodes_lifetime": 42}}, 0.1)
        assert result.episodes_total == 42

    def test_episode_len_from_new_stack(self):
        result = TrainResult.from_rllib(1, {"env_runners": {"episode_len_mean": 25.0}}, 0.1)
        assert result.episode_len_mean == pytest.approx(25.0)

    def test_extracts_rllib_timing_breakdown(self):
        result = TrainResult.from_rllib(1, {
            "timers": {
                "env_runner_sampling_timer": 1.5,
                "learner_update_timer": 2.5,
                "synch_weights": 0.25,
            },
            "num_training_step_calls_per_iteration": np.int64(10),
            "env_runners": {
                "env_step_timer": 0.4,
                "rlmodule_inference_timer": 0.3,
            },
        }, 5.0)

        assert result.timings.sampling_ema_s == pytest.approx(1.5)
        assert result.timings.learner_update_ema_s == pytest.approx(2.5)
        assert result.timings.sync_weights_ema_s == pytest.approx(0.25)
        assert result.timings.env_step_ema_s == pytest.approx(0.4)
        assert result.timings.inference_ema_s == pytest.approx(0.3)
        assert result.timings.training_step_calls == 10

    def test_missing_timing_fields_are_omitted_from_tensorboard(self):
        timings = IterationTimings(sampling_ema_s=1.0)

        scalars = timings.tensorboard_scalars(elapsed_s=2.0)

        assert scalars["performance/sampling_ema_s"] == pytest.approx(1.0)
        assert "performance/learner_update_ema_s" not in scalars
        assert scalars["performance/rllib_wall_time_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAlgo:
    """Minimal duck-typed Algorithm for standalone tests."""
    def __init__(
        self,
        reward_sequence: list[float] | None = None,
        evaluation_count: int = 1,
    ) -> None:
        self._step = 0
        self._rewards = reward_sequence or [1.0, 2.0, 3.0, 4.0, 5.0]
        self._evaluation_count = evaluation_count
        self.evaluation_episodes: list[VoxelEpisodeData] | None = None
        self.saved_paths: list[str] = []
        self.restored_paths: list[str] = []

    def train(self) -> dict[str, Any]:
        r = self._rewards[self._step % len(self._rewards)]
        self._step += 1
        episode = VoxelEpisodeData(
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
        self._anysearch_evaluation_episodes = (
            self.evaluation_episodes
            if self.evaluation_episodes is not None
            else [episode for _ in range(self._evaluation_count)]
        )
        return {
            "episode_reward_mean": r,
            "episode_len_mean": 20.0,
            "episodes_total": self._step * 3,
        }

    def save(self, path: str) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "weights.sentinel").write_text("ok")
        self.saved_paths.append(path)
        return path

    def restore(self, path: str) -> None:
        sentinel = Path(path) / "weights.sentinel"
        if not sentinel.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        self.restored_paths.append(path)


class FakeSummaryWriter:
    """Test double for torch.utils.tensorboard.SummaryWriter."""

    instances: list["FakeSummaryWriter"] = []

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_calls = 0
        self.closed = False
        self.__class__.instances.append(self)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.closed = True


def make_trainer(trainer_settings: Any, rewards: list[float] | None = None) -> PPOTrainer:
    """Return a PPOTrainer whose _build_algorithm returns a FakeAlgo."""
    algo = FakeAlgo(rewards, trainer_settings.evaluation.episodes)

    class _T(PPOTrainer):
        def _build_algorithm(self) -> FakeAlgo:
            return algo

    t = _T(trainer_settings)
    t._fake_algo = algo  # expose for assertions
    return t


# ---------------------------------------------------------------------------
# 1. Execution: train() runs the correct number of iterations
# ---------------------------------------------------------------------------

class TestExecution:
    """Tests Execution."""
    def test_returns_one_result_per_iteration(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        results = t.train()
        assert len(results) == trainer_settings.training.iterations

    def test_results_are_train_result_instances(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        results = t.train()
        assert all(isinstance(r, TrainResult) for r in results)

    def test_iteration_numbers_are_sequential(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        results = t.train()
        assert [r.iteration for r in results] == list(
            range(1, trainer_settings.training.iterations + 1)
        )

    def test_evaluation_result_replay_and_checkpoint_share_one_batch(
        self,
        trainer_settings: Any,
    ):
        trainer_settings.training.iterations = 1
        trainer_settings.evaluation.episodes = 2
        trainer_settings.training.trajectory_every = 1

        def _episode(success: bool, reward: float, steps: int) -> VoxelEpisodeData:
            return VoxelEpisodeData(
                agent_count=1,
                max_steps=10,
                obs_mode="radial",
                init_filled=[],
                steps=[
                    VoxelStepData(
                        step=index,
                        action=1,
                        reward=reward / steps,
                        done=index == steps - 1,
                        cursor_x=1,
                        cursor_y=1,
                        cursor_z=index + 1,
                        voxel_count=index + 1,
                        placed=True,
                    )
                    for index in range(steps)
                ],
                total_reward=reward,
                success=success,
                start_pos=(1, 1, 1),
                goal_pos=(1, 1, steps),
            )

        episodes = [
            _episode(False, -1.0, 4),
            _episode(True, 2.0, 2),
        ]
        trainer = make_trainer(trainer_settings)
        trainer._fake_algo.evaluation_episodes = episodes
        result = trainer.train()[0]

        output_dir = Path(trainer_settings.training.output_dir)
        summary = json.loads(
            output_dir.joinpath("evaluation", "iter_000001.json").read_text()
        )
        replay = json.loads(
            output_dir.joinpath("trajectories", "iter_000001.json").read_text()
        )
        best_meta = json.loads(
            output_dir.joinpath("trajectories", "best_meta.json").read_text()
        )

        assert result.evaluation_episodes == 2
        assert result.evaluation_goals_reached == 1
        assert result.evaluation_success_rate == pytest.approx(0.5)
        assert summary["success_rate"] == pytest.approx(0.5)
        assert [item["success"] for item in summary["episodes"]] == [False, True]
        assert replay["episode"]["success"] is True
        assert best_meta["iteration"] == 1
        assert output_dir.joinpath("checkpoints", "iter_000001").is_dir()
    def test_episode_reward_mean_comes_from_algo(self, trainer_settings: Any):
        rewards = [10.0, 20.0, 30.0, 40.0]
        t = make_trainer(trainer_settings, rewards)
        results = t.train()
        assert results[0].episode_reward_mean == pytest.approx(10.0)
        assert results[1].episode_reward_mean == pytest.approx(20.0)

    def test_on_iteration_end_called_each_step(self, trainer_settings: Any):
        calls: list[int] = []

        class _T(PPOTrainer):
            def _build_algorithm(self) -> FakeAlgo:
                return FakeAlgo()
            def on_iteration_end(self, result: TrainResult) -> None:
                calls.append(result.iteration)

        t = _T(trainer_settings)
        t.train()
        assert calls == list(range(1, trainer_settings.training.iterations + 1))


# ---------------------------------------------------------------------------
# 2. Config validity: config fields reach the Trainer correctly
# ---------------------------------------------------------------------------

class TestConfigValidity:
    """Tests ConfigValidity."""
    def test_output_dir_uses_settings(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        assert t._output_dir == Path(trainer_settings.training.output_dir)

    def test_train_respects_iteration_count(self, trainer_settings: Any):
        """iterations=4 in trainer_settings → exactly 4 train() calls."""
        t = make_trainer(trainer_settings)
        t.train()
        assert t._fake_algo._step == trainer_settings.training.iterations

    def test_checkpoint_interval_triggers_at_right_steps(self, trainer_settings: Any, tmp_path: Path):
        """checkpoint_interval=2 → checkpoints at iter 2 and 4."""
        t = make_trainer(trainer_settings)
        t.train()
        ckpt_dir = t._output_dir / "checkpoints"
        saved = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir())
        assert "iter_000002" in saved
        assert "iter_000004" in saved


# ---------------------------------------------------------------------------
# 3. Output sanity: checkpoint files exist and contain expected data
# ---------------------------------------------------------------------------

class TestOutputSanity:
    """Tests OutputSanity."""
    def test_train_writes_tensorboard_scalars_under_run_dir(self, trainer_settings: Any):
        FakeSummaryWriter.instances.clear()
        with patch(
            "torch.utils.tensorboard.SummaryWriter",
            new=FakeSummaryWriter,
        ):
            t = make_trainer(trainer_settings)
            t.train()

        assert len(FakeSummaryWriter.instances) == 1
        writer = FakeSummaryWriter.instances[0]
        assert Path(writer.log_dir) == Path(trainer_settings.training.output_dir).joinpath("tensorboard")
        assert ("train/episode_reward_mean", 1.0, 1) in writer.scalars
        assert ("train/episode_reward_mean", 4.0, 4) in writer.scalars
        assert any(
            tag == "performance/anysearch_evaluation_s"
            for tag, _, _ in writer.scalars
        )
        assert any(
            tag == "performance/rllib_wall_time_s"
            for tag, _, _ in writer.scalars
        )
        assert writer.flush_calls == trainer_settings.training.iterations * 2
        assert writer.closed is True

    def test_train_writes_eval_metrics_under_run_dir(self, trainer_settings: Any):
        FakeSummaryWriter.instances.clear()
        trainer_settings.training.trajectory_every = 1
        trainer_settings.training.best_trajectory = True
        episode = VoxelEpisodeData(
            agent_count=1,
            max_steps=10,
            obs_mode="radial",
            init_filled=[],
            total_reward=9.94,
            success=True,
            start_pos=(4, 4, 4),
            goal_pos=(4, 4, 6),
            steps=[
                VoxelStepData(
                    step=0,
                    action=1,
                    reward=-0.03,
                    done=False,
                    cursor_x=4,
                    cursor_y=4,
                    cursor_z=5,
                    voxel_count=2,
                    placed=True,
                ),
                VoxelStepData(
                    step=1,
                    action=1,
                    reward=9.97,
                    done=True,
                    cursor_x=4,
                    cursor_y=4,
                    cursor_z=6,
                    voxel_count=3,
                    placed=True,
                ),
            ],
        )
        with patch(
            "torch.utils.tensorboard.SummaryWriter",
            new=FakeSummaryWriter,
        ):
            t = make_trainer(trainer_settings)
            t._fake_algo.evaluation_episodes = [
                episode for _ in range(trainer_settings.evaluation.episodes)
            ]
            t.train()

        writer = FakeSummaryWriter.instances[0]
        assert ("eval/collision_count", 0.0, 1) in writer.scalars
        assert ("eval/finish_count", 1.0, 1) in writer.scalars
        assert ("eval/finish_rate", 1.0, 1) in writer.scalars
        assert ("eval/mean_steps_on_success", 2.0, 1) in writer.scalars
        assert ("eval/goal_progress_mean", 2.0, 1) in writer.scalars

    def test_checkpoint_creates_state_json(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        ckpt = t.checkpoint()
        assert (ckpt / "state.json").exists()

    def test_state_json_has_iteration(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 3
        ckpt = t.checkpoint()
        state = json.loads((ckpt / "state.json").read_text())
        assert state["iteration"] == 3

    def test_state_json_has_episodes_total(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 1
        t._episodes_total = 42
        ckpt = t.checkpoint()
        state = json.loads((ckpt / "state.json").read_text())
        assert state["episodes_total"] == 42

    def test_latest_json_written_after_checkpoint(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 2
        t.checkpoint()
        latest_ptr = t._output_dir / "checkpoints" / "latest.json"
        assert latest_ptr.exists()
        info = json.loads(latest_ptr.read_text())
        assert info["iteration"] == 2

    def test_latest_json_path_points_to_checkpoint(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 2
        ckpt = t.checkpoint()
        latest_ptr = t._output_dir / "checkpoints" / "latest.json"
        info = json.loads(latest_ptr.read_text())
        assert Path(info["path"]) == ckpt

    def test_weights_sentinel_written_by_algo(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 1
        ckpt = t.checkpoint()
        assert (ckpt / "weights.sentinel").exists()

    def test_train_creates_checkpoint_dirs(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t.train()
        ckpt_dir = t._output_dir / "checkpoints"
        assert ckpt_dir.is_dir()
        assert any(ckpt_dir.iterdir())


# ---------------------------------------------------------------------------
# 4. Restore: load from a specific checkpoint dir
# ---------------------------------------------------------------------------

class TestRestore:
    """Tests Restore."""
    def test_restore_sets_iteration(self, trainer_settings: Any):
        # Write a checkpoint manually
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 7
        ckpt = t.checkpoint()

        # Fresh trainer restores from it
        t2 = make_trainer(trainer_settings)
        t2.restore(ckpt)
        assert t2._iteration == 7

    def test_restore_sets_episodes_total(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 3
        t._episodes_total = 55
        ckpt = t.checkpoint()

        t2 = make_trainer(trainer_settings)
        t2.restore(ckpt)
        assert t2._episodes_total == 55

    def test_restore_builds_algo_if_none(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 1
        ckpt = t.checkpoint()

        t2 = make_trainer(trainer_settings)
        assert t2._algo is None
        t2.restore(ckpt)
        assert t2._algo is not None

    def test_restore_calls_algo_restore(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 1
        ckpt = t.checkpoint()

        t2 = make_trainer(trainer_settings)
        t2.restore(ckpt)
        assert str(ckpt) in t2._algo.restored_paths


# ---------------------------------------------------------------------------
# 5. Resume: continue training from latest checkpoint
# ---------------------------------------------------------------------------

class TestResume:
    """Tests Resume."""
    def test_resume_returns_false_when_no_checkpoint(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        assert t.resume() is False

    def test_resume_returns_true_when_checkpoint_exists(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 2
        t.checkpoint()

        t2 = make_trainer(trainer_settings)
        assert t2.resume() is True

    def test_resume_restores_correct_iteration(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 3
        t.checkpoint()

        t2 = make_trainer(trainer_settings)
        t2.resume()
        assert t2._iteration == 3

    def test_resumed_train_completes_remaining_iters(self, trainer_settings: Any):
        """After resuming from iter 2, train() runs the remaining 2 iters (total=4)."""
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 2
        t.checkpoint()

        t2 = make_trainer(trainer_settings)
        t2.resume()
        results = t2.train()
        assert len(results) == trainer_settings.training.iterations - 2

    def test_resumed_iterations_start_from_restored_point(self, trainer_settings: Any):
        t = make_trainer(trainer_settings)
        t._algo = FakeAlgo()
        t._iteration = 2
        t.checkpoint()

        t2 = make_trainer(trainer_settings)
        t2.resume()
        results = t2.train()
        assert results[0].iteration == 3


# ---------------------------------------------------------------------------
# 6. Trainer registry: all algorithms register themselves on import
# ---------------------------------------------------------------------------

class TestTrainerRegistry:
    """Tests TrainerRegistry."""
    def test_all_discrete_algorithms_registered(self):
        from theseo_anysearch.rllib.algorithms.registry import registered_algorithms

        available = registered_algorithms()
        for name in ("ppo", "appo", "sac", "dqn", "rainbow"):
            assert name in available, f"'{name}' not in the algorithm registry"

    def test_all_continuous_stubs_registered(self):
        from theseo_anysearch.rllib.algorithms.registry import registered_algorithms

        available = registered_algorithms()
        for name in ("td3", "ddpg"):
            assert name in available, f"'{name}' not in the algorithm registry"
    def test_from_settings_dispatches_to_ppo(self, trainer_settings):
        from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer
        trainer = Trainer.from_settings(trainer_settings)
        assert isinstance(trainer, PPOTrainer)

    def test_td3_raises_not_implemented(self, trainer_settings):
        from theseo_anysearch.rllib.algorithms.td3 import TD3Trainer
        trainer = TD3Trainer(trainer_settings)
        with pytest.raises(NotImplementedError, match="Box"):
            trainer._build_algorithm()

    def test_ddpg_raises_not_implemented(self, trainer_settings):
        from theseo_anysearch.rllib.algorithms.ddpg import DDPGTrainer
        trainer = DDPGTrainer(trainer_settings)
        with pytest.raises(NotImplementedError, match="Box"):
            trainer._build_algorithm()

class TestTrainingEarlyStop:
    """Standard training stops on deterministic evaluation conditions."""

    def test_goal_finish_condition_saves_final_artifacts(
        self,
        trainer_settings: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from theseo_anysearch.models import TrainingEarlyStopConfig

        episode = VoxelEpisodeData(
            agent_count=1,
            max_steps=5,
            obs_mode="scalar",
            grid_size=8,
            init_filled=[],
            steps=[
                VoxelStepData(
                    step=0,
                    action=13,
                    reward=2.0,
                    done=True,
                    cursor_x=1,
                    cursor_y=1,
                    cursor_z=2,
                    voxel_count=0,
                    placed=False,
                )
            ],
            total_reward=2.0,
            success=True,
            start_pos=(1, 1, 1),
            goal_pos=(1, 1, 2),
        )
        trainer_settings.training = trainer_settings.training.model_copy(
            update={
                "iterations": 10,
                "trajectory_every": 0,
                "best_trajectory": False,
                "early_stop": TrainingEarlyStopConfig(
                    enabled=True,
                    mode="goal_finishes",
                    min_goal_finishes=2,
                    min_consecutive_evaluation=2,
                ),
            }
        )
        trainer_settings.evaluation = trainer_settings.evaluation.model_copy(
            update={"episodes": 2}
        )
        trainer = make_trainer(trainer_settings)
        trainer._fake_algo.evaluation_episodes = [episode, episode]

        results = trainer.train()

        assert len(results) == 2
        assert trainer._iteration == 2
        assert trainer._output_dir.joinpath("early_stop.json").exists()
        assert trainer._output_dir.joinpath("checkpoints", "iter_000002").exists()
        assert trainer._output_dir.joinpath(
            "trajectories", "iter_000002.json"
        ).exists()
        payload = json.loads(
            trainer._output_dir.joinpath("early_stop.json").read_text()
        )
        assert payload["mode"] == "goal_finishes"
        assert payload["value"] == 2.0
        assert payload["iteration"] == 2
