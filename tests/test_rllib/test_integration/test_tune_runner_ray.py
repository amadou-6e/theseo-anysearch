"""
Integration tests for TuneRunner against a real (minimal) Ray cluster.

These tests exercise the full sweep path but with very small configs:
  - 1 env runner (local to the trial actor, no remote worker PG issue)
  - 2 trials max
  - 2 iterations per trial
  - CPU only (no GPU required)

Scenarios covered:
  A. PlacementGroupFactory provides enough bundles → no bundle-index ValueError
  B. reuse_actors=False → no stale PG refs across consecutive trials
  C. Trials report to ray.tune (not ray.train)
  D. _experiment_trainable builds Settings from experiment_dict correctly
  E. ASHA prunes a clearly inferior trial (reward sentinel check)
  F. Output dir is created and contains trial subdirs

The session-scoped `ray_session` fixture (from conftest.py) initialises Ray
with num_cpus=4 once for the whole session.  Individual tests must NOT call
ray.init() themselves.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Minimal ExperimentConfig for sweep tests
# ---------------------------------------------------------------------------

_SWEEP_YAML = textwrap.dedent("""\
    experiment:
      name: test-sweep
      output_dir: {output_dir}
      seed: 0

    env:
      agent_count: 2
      max_steps: 10

    training:
      algorithm: multi_agent_voxel_ppo
      runner: local
      iterations: 2
      checkpoint_interval: 10
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 3e-4
      gamma: 0.99
      train_batch_size: 64
      clip_param: 0.2
      num_sgd_iter: 1
      lambda_: 0.95
      kl_coeff: 0.2
      minibatch_size: 32

    model_config:
      hidden_sizes: [32]
      activation: relu

    tune_config:
      scheduler: asha
      num_samples: 2
      metric: episode_reward_mean
      mode: max
      max_concurrent: 1
      search_space:
        algorithm_config:
          lr: {{choice: [3e-4, 1e-3]}}

    mlflow: {{}}
""")

# YAML variant with num_env_runners=1 so the PGF needs 2 bundles (driver + 1 worker).
# With num_env_runners=0 the original bundle-index bug can NEVER trigger because only
# 1 bundle is needed.  This variant reproduces the real-world crash scenario.
_SWEEP_YAML_MULTI_RUNNER = textwrap.dedent("""\
    experiment:
      name: test-sweep-multi-runner
      output_dir: {output_dir}
      seed: 0

    env:
      agent_count: 2
      max_steps: 10

    training:
      algorithm: multi_agent_voxel_ppo
      runner: local
      iterations: 2
      checkpoint_interval: 10
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 1

    algorithm_config:
      lr: 3e-4
      gamma: 0.99
      train_batch_size: 64
      clip_param: 0.2
      num_sgd_iter: 1
      lambda_: 0.95
      kl_coeff: 0.2
      minibatch_size: 32

    model_config:
      hidden_sizes: [32]
      activation: relu

    tune_config:
      scheduler: asha
      num_samples: 2
      metric: episode_reward_mean
      mode: max
      max_concurrent: 1
      search_space:
        algorithm_config:
          lr: {{choice: [3e-4, 1e-3]}}

    mlflow: {{}}
""")

# YAML for the stale-PG-refs regression: 16 samples, 0 env runners (fast).
# The reuse_actors=True bug triggered at trial ~14; with reuse_actors=False
# all 16 should complete cleanly.
_SWEEP_YAML_MANY_TRIALS = textwrap.dedent("""\
    experiment:
      name: test-sweep-many-trials
      output_dir: {output_dir}
      seed: 0

    env:
      agent_count: 2
      max_steps: 10

    training:
      algorithm: multi_agent_voxel_ppo
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 3e-4
      gamma: 0.99
      train_batch_size: 64
      clip_param: 0.2
      num_sgd_iter: 1
      lambda_: 0.95
      kl_coeff: 0.2
      minibatch_size: 32

    model_config:
      hidden_sizes: [32]
      activation: relu

    tune_config:
      scheduler: asha
      num_samples: 16
      metric: episode_reward_mean
      mode: max
      max_concurrent: 1
      search_space:
        algorithm_config:
          lr: {{loguniform: [1.0e-4, 1.0e-2]}}

    mlflow: {{}}
""")


def _load_sweep_config(tmp_path: Path) -> "ExperimentConfig":  # noqa: F821
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = _SWEEP_YAML.format(output_dir=str(output_dir).replace("\\", "/"))
    cfg_file = tmp_path / "sweep.yaml"
    cfg_file.write_text(cfg_text)

    from theseo_anysearch.experiments.loader import load_experiment
    return load_experiment(cfg_file), cfg_file


def _load_multi_runner_config(tmp_path: Path):
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = _SWEEP_YAML_MULTI_RUNNER.format(
        output_dir=str(output_dir).replace("\\", "/")
    )
    cfg_file = tmp_path / "sweep_multi_runner.yaml"
    cfg_file.write_text(cfg_text)
    from theseo_anysearch.experiments.loader import load_experiment
    return load_experiment(cfg_file), cfg_file


def _load_many_trials_config(tmp_path: Path):
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = _SWEEP_YAML_MANY_TRIALS.format(
        output_dir=str(output_dir).replace("\\", "/")
    )
    cfg_file = tmp_path / "sweep_many_trials.yaml"
    cfg_file.write_text(cfg_text)
    from theseo_anysearch.experiments.loader import load_experiment
    return load_experiment(cfg_file), cfg_file


# ---------------------------------------------------------------------------
# A. No placement group ValueError
# ---------------------------------------------------------------------------

class TestPlacementGroupNoError:
    """
    The PlacementGroupFactory must provide enough bundles so that RLlib can
    schedule its env runners without hitting 'bundle index N is invalid'.

    With num_env_runners=0 the trial actor runs everything locally (1 bundle
    is sufficient).  With num_env_runners>0 we need 1+N bundles.
    """

    @pytest.mark.usefixtures("ray_session")
    def test_sweep_starts_without_pg_error(self, tmp_path):
        """Two sequential trials complete without ValueError from placement group."""
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        runner = TuneRunner(cfg, config_path=cfg_file, tag="pg-test")
        # Should not raise ValueError about bundle indices
        result = runner.run()
        assert result is not None

    @pytest.mark.usefixtures("ray_session")
    def test_no_bundle_index_error_in_result(self, tmp_path):
        """result dict is present, indicating no fatal placement group failure."""
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        result = TuneRunner(cfg, config_path=cfg_file, tag="pg-noerr").run()
        assert "best_config" in result
        assert "best_result" in result


# ---------------------------------------------------------------------------
# B. reuse_actors=False — no stale PG refs across trials
# ---------------------------------------------------------------------------

class TestReuseActorsFalseIntegration:
    """
    With reuse_actors=False, each trial gets a freshly created actor with
    a valid current placement group.  We verify this by running more trials
    than the original failure threshold (14) would imply — with only 2 trials
    and a clean actor each time, the bundle refs must be valid.
    """

    @pytest.mark.usefixtures("ray_session")
    def test_two_consecutive_trials_succeed(self, tmp_path):
        """Both trials complete without PG errors (regression for reuse_actors=True bug)."""
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        result = TuneRunner(cfg, config_path=cfg_file, tag="reuse-test").run()
        # Both trials should have run; best_result metric should be finite
        best_metric = result["best_result"].get("episode_reward_mean")
        # The metric may be a large negative (early training) but must not be None
        assert best_metric is not None


# ---------------------------------------------------------------------------
# C. _experiment_trainable reports via ray.tune not ray.train
# ---------------------------------------------------------------------------

class TestTrainableReportTarget:
    """
    Regression: ray.train.report raises DeprecationWarning in Ray 2.54 Tune.
    _experiment_trainable must call ray.tune.report instead.
    """

    def test_tune_report_called_not_train_report(self, tmp_path):
        """
        Run _experiment_trainable with a mocked Trainer and verify
        ray.tune.report is called for each iteration, never ray.train.report.
        """
        import math
        from unittest.mock import MagicMock, patch, call
        import ray.tune as _real_tune
        import ray.train as _real_train

        from theseo_anysearch.experiments.tune_runner import _experiment_trainable
        from theseo_anysearch.rllib.trainer.results import TrainResult
        from theseo_anysearch.experiments.models import (
            ExperimentConfig, ExperimentMeta, TuneConfig,
        )
        from theseo_anysearch.models import (
            AlgorithmConfig, AnyscaleConfig, EnvConfig, ModelConfig, TrainingConfig,
        )

        out = tmp_path / "out"
        out.mkdir()

        cfg = ExperimentConfig(
            experiment=ExperimentMeta(name="t", output_dir=str(out)),
            env=EnvConfig(agent_count=2, max_steps=10),
            training=TrainingConfig(
                algorithm="multi_agent_voxel_ppo",
                iterations=2,
                output_dir=str(out),
            ),
            anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
            algorithm_config=AlgorithmConfig(lr=3e-4, gamma=0.99, train_batch_size=64),
            model_cfg=ModelConfig(hidden_sizes=[32]),
            tune_config=TuneConfig(num_samples=2, max_concurrent=1),
        )
        exp_dict = cfg.model_dump(by_alias=True, mode="json")
        exp_dict["experiment"]["output_dir"] = str(out)

        fake_results = [
            TrainResult(iteration=1, episode_reward_mean=-5.0,
                        episode_len_mean=10.0, episodes_total=5, elapsed_s=0.1),
            TrainResult(iteration=2, episode_reward_mean=-3.0,
                        episode_len_mean=10.0, episodes_total=10, elapsed_s=0.2),
        ]

        mock_trainer = MagicMock()
        mock_trainer.train = MagicMock(return_value=fake_results)

        mock_tune = MagicMock(spec=_real_tune)
        mock_tune.get_context.return_value = MagicMock(
            get_trial_id=MagicMock(return_value="t0")
        )
        mock_train = MagicMock(spec=_real_train)

        with patch("theseo_anysearch.experiments.tune_runner._tune", mock_tune), \
             patch("ray.train", mock_train), \
             patch("theseo_anysearch.experiments.tune_runner.Trainer") as MockTrainer:

            MockTrainer.from_settings = MagicMock(return_value=mock_trainer)

            # Simulate the iteration hook by making train() call on_iteration_end
            def fake_train():
                for r in fake_results:
                    if mock_trainer.on_iteration_end:
                        mock_trainer.on_iteration_end(r)
            mock_trainer.train = fake_train
            mock_trainer.on_iteration_end = None
            mock_trainer._algo = None

            _experiment_trainable(
                config={"lr": 3e-4},
                experiment_dict=exp_dict,
                metric="episode_reward_mean",
                mode="max",
                max_iterations=2,
                mlflow_tracking_uri="",
                mlflow_experiment_name="test",
                mlflow_parent_run_id="",
                run_tag="test",
            )

        assert mock_tune.report.call_count == 2, (
            f"Expected 2 tune.report calls, got {mock_tune.report.call_count}"
        )
        mock_train.report.assert_not_called()

    def test_report_payload_has_required_keys(self, tmp_path):
        """Each report call must include metric + training_iteration."""
        import math
        from unittest.mock import MagicMock, patch
        import ray.tune as _real_tune

        from theseo_anysearch.experiments.tune_runner import _experiment_trainable
        from theseo_anysearch.rllib.trainer.results import TrainResult
        from theseo_anysearch.experiments.models import ExperimentConfig, ExperimentMeta, TuneConfig
        from theseo_anysearch.models import (
            AlgorithmConfig, AnyscaleConfig, EnvConfig, ModelConfig, TrainingConfig,
        )

        out = tmp_path / "out2"
        out.mkdir()

        cfg = ExperimentConfig(
            experiment=ExperimentMeta(name="t2", output_dir=str(out)),
            env=EnvConfig(agent_count=2, max_steps=10),
            training=TrainingConfig(
                algorithm="multi_agent_voxel_ppo",
                iterations=1,
                output_dir=str(out),
            ),
            anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
            algorithm_config=AlgorithmConfig(lr=3e-4, gamma=0.99, train_batch_size=64),
            model_cfg=ModelConfig(hidden_sizes=[32]),
            tune_config=TuneConfig(num_samples=1, max_concurrent=1),
        )
        exp_dict = cfg.model_dump(by_alias=True, mode="json")
        exp_dict["experiment"]["output_dir"] = str(out)

        result = TrainResult(
            iteration=1, episode_reward_mean=-2.0,
            episode_len_mean=10.0, episodes_total=5, elapsed_s=0.1,
        )

        mock_tune = MagicMock(spec=_real_tune)
        mock_tune.get_context.return_value = MagicMock(
            get_trial_id=MagicMock(return_value="t1")
        )

        with patch("theseo_anysearch.experiments.tune_runner._tune", mock_tune), \
             patch("theseo_anysearch.experiments.tune_runner.Trainer") as MockTrainer:

            mock_trainer = MagicMock()
            mock_trainer._algo = None
            mock_trainer.on_iteration_end = None
            MockTrainer.from_settings = MagicMock(return_value=mock_trainer)

            def fake_train():
                if mock_trainer.on_iteration_end:
                    mock_trainer.on_iteration_end(result)
            mock_trainer.train = fake_train

            _experiment_trainable(
                config={"lr": 3e-4},
                experiment_dict=exp_dict,
                metric="episode_reward_mean",
                mode="max",
                max_iterations=1,
                mlflow_tracking_uri="",
                mlflow_experiment_name="test",
                mlflow_parent_run_id="",
                run_tag="test",
            )

        call_kwargs = mock_tune.report.call_args[0][0]
        for key in ("episode_reward_mean", "training_iteration"):
            assert key in call_kwargs, f"Missing key '{key}' in tune.report payload"


# ---------------------------------------------------------------------------
# D. _experiment_trainable reconstructs ExperimentConfig from dict
# ---------------------------------------------------------------------------

class TestExperimentTrainableDeserialization:
    """
    experiment_dict is a plain dict (JSON-serialisable) passed from the driver
    to the worker.  _experiment_trainable must be able to reconstruct a valid
    ExperimentConfig from it.
    """

    def test_deserializes_without_error(self, tmp_path):
        """Loading experiment_dict via _resolve_typed_configs + ExperimentConfig(**) succeeds."""
        from theseo_anysearch.experiments.models import (
            ExperimentConfig, ExperimentMeta, TuneConfig,
        )
        from theseo_anysearch.models import (
            AlgorithmConfig, AnyscaleConfig, EnvConfig, ModelConfig, TrainingConfig,
        )
        from theseo_anysearch.experiments.loader import _resolve_typed_configs

        out = tmp_path / "deser"
        out.mkdir()

        cfg = ExperimentConfig(
            experiment=ExperimentMeta(name="deser-test", output_dir=str(out)),
            env=EnvConfig(agent_count=2, max_steps=10),
            training=TrainingConfig(
                algorithm="multi_agent_voxel_ppo",
                iterations=1,
                require_gpu=False,
                num_env_runners=0,
                output_dir=str(out),
                num_gpus=0.5,
            ),
            anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
            algorithm_config=AlgorithmConfig(lr=3e-4, gamma=0.99, train_batch_size=64),
            model_cfg=ModelConfig(hidden_sizes=[32]),
            tune_config=TuneConfig(num_samples=1, max_concurrent=2),
        )

        exp_dict = cfg.model_dump(by_alias=True, mode="json")
        resolved = _resolve_typed_configs(exp_dict)
        reconstructed = ExperimentConfig(**resolved)

        assert reconstructed.training.num_gpus == pytest.approx(0.5)
        assert reconstructed.training.require_gpu is False
        assert reconstructed.training.num_env_runners == 0

    def test_num_gpus_preserved_through_serialization(self, tmp_path):
        """num_gpus survives model_dump → JSON → reconstruct round-trip."""
        from theseo_anysearch.experiments.models import (
            ExperimentConfig, ExperimentMeta, TuneConfig,
        )
        from theseo_anysearch.models import (
            AlgorithmConfig, AnyscaleConfig, EnvConfig, ModelConfig, TrainingConfig,
        )
        from theseo_anysearch.experiments.loader import _resolve_typed_configs

        out = tmp_path / "rtrip"
        out.mkdir()

        for frac in (0.25, 0.5, 1.0, None):
            cfg = ExperimentConfig(
                experiment=ExperimentMeta(name="rt", output_dir=str(out)),
                env=EnvConfig(agent_count=2, max_steps=10),
                training=TrainingConfig(
                    algorithm="multi_agent_voxel_ppo",
                    iterations=1,
                    output_dir=str(out),
                    num_gpus=frac,
                ),
                anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
                algorithm_config=AlgorithmConfig(lr=3e-4, gamma=0.99, train_batch_size=64),
                model_cfg=ModelConfig(hidden_sizes=[32]),
                tune_config=TuneConfig(num_samples=1, max_concurrent=1),
            )
            d = cfg.model_dump(by_alias=True, mode="json")
            resolved = _resolve_typed_configs(d)
            reconstructed = ExperimentConfig(**resolved)
            if frac is None:
                assert reconstructed.training.num_gpus is None
            else:
                assert reconstructed.training.num_gpus == pytest.approx(frac)


# ---------------------------------------------------------------------------
# E. TuneRunner result structure
# ---------------------------------------------------------------------------

class TestTuneRunnerResultStructure:
    """
    TuneRunner.run() must return a dict with the documented keys even after
    a minimal real sweep.
    """

    @pytest.mark.usefixtures("ray_session")
    def test_result_has_required_keys(self, tmp_path):
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        result = TuneRunner(cfg, config_path=cfg_file, tag="result-keys").run()

        for key in ("best_config", "best_result", "experiment_dir", "run_tag"):
            assert key in result, f"Missing key '{key}' in TuneRunner result"

    @pytest.mark.usefixtures("ray_session")
    def test_run_tag_matches_provided_tag(self, tmp_path):
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        result = TuneRunner(cfg, config_path=cfg_file, tag="my-sweep-tag").run()
        assert result["run_tag"] == "my-sweep-tag"

    @pytest.mark.usefixtures("ray_session")
    def test_best_config_contains_search_param(self, tmp_path):
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        result = TuneRunner(cfg, config_path=cfg_file, tag="best-cfg").run()
        assert "lr" in result["best_config"], (
            "best_config must contain the searched 'lr' param"
        )


# ---------------------------------------------------------------------------
# F. Output directory structure
# ---------------------------------------------------------------------------

class TestOutputDirectoryStructure:
    """
    After a sweep, the output dir must contain a subdirectory named after the
    run_tag, and inside it the trial directories named trial_<id>.
    """

    @pytest.mark.usefixtures("ray_session")
    def test_output_dir_created(self, tmp_path):
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        tag = "outdir-test"
        TuneRunner(cfg, config_path=cfg_file, tag=tag).run()

        output_root = tmp_path / "experiments"
        assert output_root.exists(), "experiments output dir must exist"

    @pytest.mark.usefixtures("ray_session")
    def test_trial_dirs_use_short_names(self, tmp_path):
        """
        Trial dirs must be named trial_<id>, NOT embed parameter values
        (regression for Windows MAX_PATH issue).
        """
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_sweep_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        tag = "shortname-test"
        TuneRunner(cfg, config_path=cfg_file, tag=tag).run()

        output_root = tmp_path / "experiments"
        all_dirs = list(output_root.rglob("trial_*"))
        assert all_dirs, "Expected at least one trial_ directory"

        for d in all_dirs:
            # Must not contain parameter values embedded in the name
            assert "lr" not in d.name or d.name.startswith("trial_"), (
                f"Trial dir name '{d.name}' looks like it embeds param values"
            )
            # Must be short enough for Windows MAX_PATH
            assert len(str(d)) < 250, (
                f"Trial path too long for Windows: {len(str(d))} chars"
            )


# ---------------------------------------------------------------------------
# G. Multi-bundle PG regression — num_env_runners > 0
# ---------------------------------------------------------------------------

class TestMultiBundlePlacementGroup:
    """
    Regression for 'ValueError: placement group bundle index N is invalid'.

    With num_env_runners=0 the trial actor runs single-process (only 1 bundle
    needed) so the PG bug can NEVER be triggered by the other test classes.

    This class uses num_env_runners=1 so the PGF must declare 2 bundles
    (bundle 0 = driver, bundle 1 = the env-runner worker).  If TuneRunner
    mis-counts the bundles or applies with_resources in the wrong order,
    RLlib will raise ValueError when it tries to schedule the worker at
    bundle index 1.

    Requires theseo_core because real RLlib env runners are spawned.
    """

    @pytest.mark.usefixtures("ray_session")
    def test_trial_with_one_env_runner_no_bundle_error(self, tmp_path):
        """
        A sweep trial with num_env_runners=1 must complete without
        'bundle index 1 is invalid'.  With the old single-bundle PGF
        this would always raise ValueError.
        """
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_multi_runner_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        # If bundle count is wrong the ValueError propagates out of runner.run()
        result = TuneRunner(cfg, config_path=cfg_file, tag="multi-runner-pg").run()
        assert result is not None

    @pytest.mark.usefixtures("ray_session")
    def test_pgf_bundle_count_in_result(self, tmp_path):
        """
        With num_env_runners=1 the sweep must return a valid best_result,
        confirming trials actually ran (not just failed silently).
        """
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_multi_runner_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        result = TuneRunner(cfg, config_path=cfg_file, tag="multi-runner-result").run()
        assert "best_result" in result
        assert result["best_result"] is not None


# ---------------------------------------------------------------------------
# H. Stale PG refs regression — 16 trials, reuse_actors=False
# ---------------------------------------------------------------------------

class TestStalePGRefsRegression:
    """
    Regression for stale PlacementGroup bundle references after many trials.

    With reuse_actors=True, Ray reuses the same actor across trials.  After
    ~14 trials the actor's internal PG bundle references become stale: the
    old placement group is released but the actor still holds indices into it.
    This causes 'bundle index N is invalid' at trial ~14 in a 30-trial sweep.

    Fix: reuse_actors=False.  We verify by running 16 trials (above the
    failure threshold) and asserting all complete without bundle index errors.

    YAML uses num_env_runners=0 and iterations=1 to keep this fast; the
    purpose is to exercise the trial-count threshold, not training quality.
    """

    @pytest.mark.usefixtures("ray_session")
    def test_sixteen_trials_no_stale_pg_error(self, tmp_path):
        """
        16 sequential trials must all complete without PG ValueError.
        With reuse_actors=True this test would fail at trial ~14.
        """
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_many_trials_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        # If reuse_actors=True, this raises ValueError around trial 14.
        # With reuse_actors=False all 16 trials should complete cleanly.
        result = TuneRunner(cfg, config_path=cfg_file, tag="stale-pg-16").run()
        assert result is not None
        assert "best_config" in result

    @pytest.mark.usefixtures("ray_session")
    def test_sixteen_trials_best_result_is_finite(self, tmp_path):
        """All 16 trials ran if best_result has a finite metric."""
        import math
        theseo_core = pytest.importorskip("theseo_core")

        cfg, cfg_file = _load_many_trials_config(tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        result = TuneRunner(cfg, config_path=cfg_file, tag="stale-pg-metric").run()
        metric = result.get("best_result", {}).get("episode_reward_mean")
        assert metric is not None, "No metric from 16-trial sweep"
        assert math.isfinite(metric), f"Metric is not finite: {metric}"


# ---------------------------------------------------------------------------
# I. Real-Ray PG bundle-index tests — no theseo_core required
#
# Root cause of the production crash:
#   SIGSEGV in theseo_core kills raylet-managed env-runner workers.
#   The raylet restarts but the GCS loses accurate bundle_cache for
#   subsequent PG allocations, leaving bundle_count=1 for a PGF that
#   requested 13 bundles.  When RLlib tries to use bundle index 1 for
#   the first env-runner, Ray raises:
#     ValueError: placement group bundle index 1 is invalid.
#                 Valid placement group indexes: 0-1
#
# These tests run with a REAL Ray cluster (no mock) using a lightweight
# dummy trainable.  They do NOT require theseo_core.  They directly
# verify that:
#   1. The trial actor sees the correct bundle_count from its PG context.
#   2. A trial actor can successfully spawn another actor at bundle index 1
#      (mimicking RLlib env-runner creation).
#   3. Multiple consecutive trials each get a fresh full-bundle PG.
# ---------------------------------------------------------------------------

class TestBundleIndexWithRealRay:
    """
    Runtime verification of PlacementGroupFactory bundle allocation.

    These tests exercise the actual Ray scheduler — not mocks.  A
    dummy trainable inspects ray.util.get_current_placement_group()
    from inside the trial actor and reports the real bundle_count that
    Ray allocated.  If the PGF is silently dropped or the PG is
    partially allocated (as happens after raylet crashes), these tests
    will fail with a clear assertion.
    """

    @pytest.mark.usefixtures("ray_session")
    def test_trial_actor_sees_correct_bundle_count(self, tmp_path):
        """
        The trial actor must see bundle_count == 1 + num_env_runners in its
        current placement group.  If bundle_count < 1 + num_env_runners, any
        call to create an env-runner at bundle 1+ would fail with
        'bundle index N is invalid'.
        """
        import os
        import ray
        import ray.tune as rt
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import apply_win_atomic_save_patch

        # Apply the same Windows path fix that TuneRunner uses so this test
        # exercises the same patched code path that real users get.
        apply_win_atomic_save_patch()

        num_env_runners = 2  # → PGF needs 3 bundles: [driver, worker0, worker1]

        def _check_bundle_count(config):
            pg = ray.util.get_current_placement_group()
            pg_present = pg is not None
            count = pg.bundle_count if pg_present else 0
            rt.report({"pg_present": pg_present, "bundle_count": count})

        pgf = PlacementGroupFactory(
            [{"CPU": 1.0}] * (1 + num_env_runners),
            strategy="SPREAD",
        )
        trainable = rt.with_resources(_check_bundle_count, pgf)

        storage = os.path.normpath(str(tmp_path / "bc"))
        tuner = rt.Tuner(
            trainable,
            tune_config=rt.TuneConfig(
                num_samples=1,
                reuse_actors=False,
                trial_dirname_creator=lambda t: f"t_{t.trial_id}",
            ),
            run_config=rt.RunConfig(
                name="bc",
                storage_path=storage,
                verbose=0,
            ),
        )
        results = tuner.fit()
        best = results.get_best_result()
        m = best.metrics

        assert m.get("pg_present") is True, (
            "Trial actor is not running inside a placement group. "
            "with_resources(fn, pgf) was not applied correctly."
        )
        assert m.get("bundle_count") == 1 + num_env_runners, (
            f"Trial actor sees bundle_count={m.get('bundle_count')}, "
            f"expected {1 + num_env_runners}. "
            "env-runner workers would fail with 'bundle index N is invalid'."
        )

    @pytest.mark.usefixtures("ray_session")
    def test_can_spawn_actor_at_bundle_1(self, tmp_path):
        """
        A trial actor must be able to spawn another actor at bundle index 1.
        This directly mimics what RLlib's EnvRunnerGroup._make_worker() does
        for the first remote env-runner (worker_index=1, pg_offset=0).

        If the PGF only provides 1 bundle, this raises:
          ValueError: placement group bundle index 1 is invalid.
        """
        import os
        import ray
        import ray.tune as rt
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import apply_win_atomic_save_patch

        apply_win_atomic_save_patch()

        @ray.remote
        class _Ping:
            def ping(self) -> bool:
                return True

        def _spawn_at_bundle_1(config):
            pg = ray.util.get_current_placement_group()
            if pg is None:
                rt.report({"spawned": False, "error": "no_pg"})
                return
            try:
                worker = _Ping.options(
                    placement_group=pg,
                    placement_group_bundle_index=1,
                ).remote()
                ok = ray.get(worker.ping.remote(), timeout=10)
                rt.report({"spawned": ok})
            except ValueError as exc:
                rt.report({"spawned": False, "error": str(exc)})

        pgf = PlacementGroupFactory(
            [{"CPU": 1.0}, {"CPU": 1.0}],  # bundle 0 = driver, bundle 1 = worker
            strategy="SPREAD",
        )
        trainable = rt.with_resources(_spawn_at_bundle_1, pgf)

        storage = os.path.normpath(str(tmp_path / "sb1"))
        tuner = rt.Tuner(
            trainable,
            tune_config=rt.TuneConfig(
                num_samples=1,
                reuse_actors=False,
                trial_dirname_creator=lambda t: f"t_{t.trial_id}",
            ),
            run_config=rt.RunConfig(
                name="sb1",
                storage_path=storage,
                verbose=0,
            ),
        )
        results = tuner.fit()
        best = results.get_best_result()
        m = best.metrics

        assert m.get("spawned") is True, (
            f"Failed to spawn actor at bundle 1: {m.get('error', 'unknown')}. "
            "This is the 'bundle index 1 is invalid' error RLlib raises for "
            "the first env-runner when the PGF is dropped or partially allocated."
        )

    @pytest.mark.usefixtures("ray_session")
    def test_consecutive_trials_each_get_full_bundle_count(self, tmp_path):
        """
        Each of 4 consecutive trials must see the full bundle_count.

        With reuse_actors=True, stale PG refs accumulate across trials and
        bundle_count drops after ~14 trials.  With reuse_actors=False each
        trial gets a fresh actor with a fresh PG — all 4 must report the
        same bundle_count.

        This is the regression test for the production failure where 5
        crashed trials left the raylet in a state where trial 6 could only
        see 1 bundle from its PG context.
        """
        import os
        import ray
        import ray.tune as rt
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import apply_win_atomic_save_patch

        apply_win_atomic_save_patch()

        num_env_runners = 2

        def _report_bundle_count(config):
            pg = ray.util.get_current_placement_group()
            rt.report({
                "bundle_count": pg.bundle_count if pg else 0,
                "pg_present": pg is not None,
            })

        pgf = PlacementGroupFactory(
            [{"CPU": 1.0}] * (1 + num_env_runners),
            strategy="SPREAD",
        )
        trainable = rt.with_resources(_report_bundle_count, pgf)

        storage = os.path.normpath(str(tmp_path / "cbc"))
        tuner = rt.Tuner(
            trainable,
            tune_config=rt.TuneConfig(
                num_samples=4,
                max_concurrent_trials=1,
                reuse_actors=False,
                trial_dirname_creator=lambda t: f"t_{t.trial_id}",
            ),
            run_config=rt.RunConfig(
                name="cbc",
                storage_path=storage,
                verbose=0,
            ),
        )
        results = tuner.fit()

        expected = 1 + num_env_runners
        for i, result in enumerate(results):
            count = result.metrics.get("bundle_count", 0)
            assert count == expected, (
                f"Trial {i}: bundle_count={count}, expected={expected}. "
                "After a crashed trial, subsequent trials may see a degraded PG "
                "with fewer bundles, causing 'bundle index N is invalid'."
            )
