"""
Unit tests for TuneRunner construction logic.

These tests do NOT start Ray.  They verify:

1. GPU fraction calculation (require_gpu × max_concurrent combinations)
2. PlacementGroupFactory bundle count and GPU inclusion
3. with_resources applied BEFORE with_parameters (ordering regression)
4. reuse_actors=False is set in TuneConfig
5. gpu_fraction injected into experiment_dict (training.num_gpus)
6. _trial_dirname produces short names (Windows MAX_PATH guard)
7. num_gpus override in _detect_num_gpus
8. _print_sweep_overview renders Categorical/Float samplers readably
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_experiment_config(
    *,
    require_gpu: bool = False,
    num_env_runners: int = 4,
    max_concurrent: int = 1,
    iterations: int = 10,
    output_dir: str = "/tmp/exp",
) -> "ExperimentConfig":  # noqa: F821
    from theseo_anysearch.experiments.models import (
        ExperimentConfig,
        ExperimentMeta,
        TuneConfig,
    )
    from theseo_anysearch.models import (
        AlgorithmConfig,
        AnyscaleConfig,
        EnvConfig,
        ModelConfig,
        TrainingConfig,
    )

    return ExperimentConfig(
        experiment=ExperimentMeta(name="test-sweep", output_dir=output_dir),
        env=EnvConfig(
            agent_count=2,
            max_steps=50,
        ),
        training=TrainingConfig(
            algorithm="multi_agent_voxel_ppo",
            iterations=iterations,
            require_gpu=require_gpu,
            num_env_runners=num_env_runners,
            output_dir=output_dir,
        ),
        anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
        algorithm_config=AlgorithmConfig(lr=3e-4, gamma=0.99, train_batch_size=512),
        model_cfg=ModelConfig(hidden_sizes=[64]),
        tune_config=TuneConfig(
            scheduler="asha",
            num_samples=4,
            max_concurrent=max_concurrent,
            metric="episode_reward_mean",
            mode="max",
        ),
    )


# ---------------------------------------------------------------------------
# 1. GPU fraction calculation
# ---------------------------------------------------------------------------

class TestGpuFraction:
    """
    TuneRunner computes gpu_fraction before serialising the config.
    Rules:
        require_gpu=False               → 0.0
        require_gpu=True, concurrent=1  → 1.0
        require_gpu=True, concurrent=2  → 0.5
        require_gpu=True, concurrent=4  → 0.25
    """

    def _fraction(self, require_gpu: bool, max_concurrent: int) -> float:
        """Mirror the logic in TuneRunner.run() without starting Ray."""
        if require_gpu and max_concurrent > 1:
            return 1.0 / max_concurrent
        elif require_gpu:
            return 1.0
        return 0.0

    def test_no_gpu(self):
        assert self._fraction(False, 1) == 0.0

    def test_no_gpu_concurrent(self):
        assert self._fraction(False, 4) == 0.0

    def test_single_trial_gpu(self):
        assert self._fraction(True, 1) == pytest.approx(1.0)

    def test_two_concurrent_gpu(self):
        assert self._fraction(True, 2) == pytest.approx(0.5)

    def test_four_concurrent_gpu(self):
        assert self._fraction(True, 4) == pytest.approx(0.25)

    def test_fractions_sum_to_at_most_one(self):
        """N concurrent trials each claiming 1/N GPU should not exceed 1 GPU."""
        for n in (1, 2, 3, 4, 8):
            fraction = self._fraction(True, n)
            assert fraction * n <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 2. PlacementGroupFactory bundle structure
# ---------------------------------------------------------------------------

class TestPlacementGroupBundles:
    """
    The placement group must have exactly 1 + num_env_runners bundles:
      bundle 0  → driver (CPU [+ GPU])
      bundles 1..N → one per env runner (CPU only)
    """

    def _build_bundles(
        self,
        num_env_runners: int,
        gpu_fraction: float,
    ) -> list[dict]:
        driver: dict[str, float] = {"CPU": 1.0}
        if gpu_fraction > 0:
            driver["GPU"] = gpu_fraction
        return [driver] + [{"CPU": 1.0}] * num_env_runners

    def test_bundle_count_8_workers(self):
        bundles = self._build_bundles(8, 0.0)
        assert len(bundles) == 9  # 1 driver + 8 workers

    def test_bundle_count_12_workers(self):
        bundles = self._build_bundles(12, 0.0)
        assert len(bundles) == 13

    def test_driver_bundle_has_gpu_when_required(self):
        bundles = self._build_bundles(4, 0.5)
        assert bundles[0].get("GPU") == pytest.approx(0.5)

    def test_driver_bundle_has_no_gpu_when_not_required(self):
        bundles = self._build_bundles(4, 0.0)
        assert "GPU" not in bundles[0]

    def test_worker_bundles_have_no_gpu(self):
        bundles = self._build_bundles(4, 1.0)
        for b in bundles[1:]:
            assert "GPU" not in b

    def test_all_bundles_have_cpu(self):
        bundles = self._build_bundles(4, 0.5)
        for b in bundles:
            assert b.get("CPU") == pytest.approx(1.0)

    def test_bundle_count_zero_workers(self):
        """num_env_runners=0 means local training; only 1 driver bundle."""
        bundles = self._build_bundles(0, 0.0)
        assert len(bundles) == 1

    @pytest.mark.parametrize("n", [1, 4, 8, 12, 16])
    def test_bundle_count_parametrize(self, n):
        bundles = self._build_bundles(n, 0.0)
        assert len(bundles) == n + 1


# ---------------------------------------------------------------------------
# 2b. PlacementGroupFactory object construction (not just list arithmetic)
# ---------------------------------------------------------------------------

class TestPlacementGroupFactoryCreation:
    """
    Actually instantiate PlacementGroupFactory objects and verify their
    bundle structure.  TestPlacementGroupBundles only checks list arithmetic;
    these tests confirm the Ray API accepts the format TuneRunner uses and
    that the bundles are preserved through PGF construction.
    """

    def test_pgf_accepts_12_runner_bundles(self):
        """PlacementGroupFactory can be instantiated with 13 bundles (1 + 12 workers)."""
        from ray.tune import PlacementGroupFactory
        driver = {"CPU": 1.0}
        workers = [{"CPU": 1.0}] * 12
        pgf = PlacementGroupFactory([driver] + workers, strategy="SPREAD")
        assert len(pgf.bundles) == 13

    def test_pgf_accepts_zero_runner_bundles(self):
        """num_env_runners=0 → single bundle is accepted by Ray."""
        from ray.tune import PlacementGroupFactory
        pgf = PlacementGroupFactory([{"CPU": 1.0}], strategy="SPREAD")
        assert len(pgf.bundles) == 1

    def test_pgf_driver_gpu_fraction_preserved(self):
        """GPU fraction in the driver bundle survives PGF construction."""
        from ray.tune import PlacementGroupFactory
        pgf = PlacementGroupFactory(
            [{"CPU": 1.0, "GPU": 0.5}, {"CPU": 1.0}],
            strategy="SPREAD",
        )
        assert pgf.bundles[0].get("GPU") == pytest.approx(0.5)
        assert "GPU" not in pgf.bundles[1]

    def test_tunerunner_passes_pgf_to_with_resources(self, tmp_path):
        """
        TuneRunner must pass a real PlacementGroupFactory (not None or a MagicMock)
        to ray_tune.with_resources.  This is the direct guard against the bug where
        with_resources received an incompatible object and silently dropped the PGF,
        leading to 'bundle index N is invalid' at runtime.
        """
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        captured_resources = []

        def fake_with_resources(fn, resources):
            captured_resources.append(resources)
            return MagicMock(name="trainable_with_resources")

        cfg = _make_experiment_config(
            require_gpu=False, num_env_runners=4, max_concurrent=1,
            output_dir=str(tmp_path),
        )

        patches = _patch_ray_tune(fake_with_resources=fake_with_resources)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured_resources, "with_resources was never called"
        resources = captured_resources[0]
        assert isinstance(resources, PlacementGroupFactory), (
            f"Expected PlacementGroupFactory, got {type(resources).__name__}. "
            "Likely with_resources ordering bug: PGF was applied after with_parameters."
        )
        assert len(resources.bundles) == 5, (  # 1 driver + 4 workers
            f"Expected 5 bundles (1 driver + 4 workers), got {len(resources.bundles)}. "
            "This would cause 'bundle index N is invalid' for env runner workers."
        )

    @pytest.mark.parametrize("num_runners", [0, 1, 4, 12])
    def test_tunerunner_pgf_bundle_count_matches_num_env_runners(self, num_runners, tmp_path):
        """
        Bundle count in the PGF passed to with_resources must equal
        1 + num_env_runners for every possible num_env_runners value.
        Off-by-one here is the exact cause of 'bundle index N is invalid'.
        """
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        captured = []

        def fake_with_resources(fn, resources):
            captured.append(resources)
            return MagicMock()

        cfg = _make_experiment_config(
            require_gpu=False, num_env_runners=num_runners, max_concurrent=1,
            output_dir=str(tmp_path),
        )

        patches = _patch_ray_tune(fake_with_resources=fake_with_resources)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured, "with_resources not called"
        pgf = captured[0]
        assert isinstance(pgf, PlacementGroupFactory)
        assert len(pgf.bundles) == 1 + num_runners, (
            f"num_env_runners={num_runners} → expected {1 + num_runners} bundles, "
            f"got {len(pgf.bundles)}"
        )


# ---------------------------------------------------------------------------
# 3. with_resources applied BEFORE with_parameters (ordering regression)
# ---------------------------------------------------------------------------

def _make_fake_tuner_fit():
    return MagicMock(
        fit=MagicMock(return_value=MagicMock(
            get_best_result=MagicMock(return_value=MagicMock(
                config={}, metrics={"episode_reward_mean": 0.0, "training_iteration": 1}
            ))
        ))
    )


def _patch_ray_tune(
    fake_with_resources=None,
    fake_with_parameters=None,
    fake_tune_config=None,
):
    """
    Return a context manager stack that patches ray.tune.* functions.
    ray and ray_tune are imported locally inside TuneRunner.run(), so we patch
    the actual module attributes (ray.init, ray.shutdown, ray.tune.*).
    """
    import ray.tune as _rt
    patches = [
        patch("ray.init"),
        patch("ray.shutdown"),
        patch.object(_rt, "with_resources", fake_with_resources or MagicMock(return_value=MagicMock())),
        patch.object(_rt, "with_parameters", fake_with_parameters or MagicMock(return_value=MagicMock())),
        patch.object(_rt, "TuneConfig", fake_tune_config or MagicMock(return_value=MagicMock())),
        patch.object(_rt, "RunConfig", MagicMock(return_value=MagicMock())),
        patch.object(_rt, "Tuner", MagicMock(return_value=_make_fake_tuner_fit())),
        patch("theseo_anysearch.cli.commands.tune._build_scheduler", return_value=MagicMock()),
        patch("theseo_anysearch.cli.commands.tune._build_search_alg", return_value=None),
        patch("theseo_anysearch.cli.commands.tune._parse_search_space", return_value={}),
        patch("theseo_anysearch.experiments.tune_runner._print_sweep_overview"),
    ]
    return patches


class TestWithResourcesOrderingRegression:
    """
    In the Ray version used here, applying with_resources AFTER with_parameters
    silently drops the PlacementGroupFactory.  TuneRunner must apply
    with_resources first.
    """

    def test_with_resources_called_before_with_parameters(self, tmp_path):
        """
        with_resources must wrap _experiment_trainable directly (the raw
        function), not the result of with_parameters.
        """
        from theseo_anysearch.experiments.tune_runner import (
            TuneRunner, _experiment_trainable,
        )
        import ray.tune as _rt

        call_order = []

        def fake_with_resources(fn, resources):
            call_order.append(("with_resources", fn))
            return MagicMock(name="trainable_with_resources")

        def fake_with_parameters(fn, **kwargs):
            call_order.append(("with_parameters", fn))
            return MagicMock(name="trainable_final")

        cfg = _make_experiment_config(
            require_gpu=False, num_env_runners=2, max_concurrent=1,
            output_dir=str(tmp_path),
        )

        patches = _patch_ray_tune(fake_with_resources, fake_with_parameters)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                TuneRunner(cfg, config_path=None, tag="test").run()
            except Exception:
                pass

        assert len(call_order) >= 2, "with_resources and with_parameters were not both called"
        assert call_order[0][0] == "with_resources", (
            f"Expected with_resources first, got {call_order[0][0]}"
        )
        assert call_order[0][1] is _experiment_trainable, (
            "with_resources must wrap _experiment_trainable directly, "
            "not the result of with_parameters"
        )
        assert call_order[1][0] == "with_parameters"


# ---------------------------------------------------------------------------
# 4. reuse_actors=False in TuneConfig
# ---------------------------------------------------------------------------

class TestReuseActorsFalse:
    """
    reuse_actors=True causes stale placement group bundle references when a
    trial actor is reused after many trials.  Must always be False.
    """

    def test_reuse_actors_is_false(self, tmp_path):
        """TuneRunner must pass reuse_actors=False to ray.tune.TuneConfig."""
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        import ray.tune as _rt

        tune_config_calls: list[dict] = []

        def fake_tune_config(**kwargs):
            tune_config_calls.append(kwargs)
            return MagicMock()

        cfg = _make_experiment_config(output_dir=str(tmp_path))

        patches = _patch_ray_tune(fake_tune_config=fake_tune_config)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert tune_config_calls, "TuneConfig was never called"
        assert tune_config_calls[0].get("reuse_actors") is False, (
            "reuse_actors must be False — True causes stale PG bundle refs"
        )


# ---------------------------------------------------------------------------
# 5. gpu_fraction injected into experiment_dict
# ---------------------------------------------------------------------------

class TestGpuFractionInjectedIntoExperimentDict:
    """
    TuneRunner must serialise training.num_gpus = gpu_fraction into the
    experiment_dict that is passed to _experiment_trainable via with_parameters.
    """

    def _run_and_capture_experiment_dict(
        self, require_gpu: bool, max_concurrent: int, tmp_path: Path
    ) -> dict:
        captured: list[dict] = []

        def fake_with_parameters(fn, **kwargs):
            captured.append(kwargs.get("experiment_dict", {}))
            return MagicMock()

        cfg = _make_experiment_config(
            require_gpu=require_gpu,
            max_concurrent=max_concurrent,
            output_dir=str(tmp_path),
        )

        from theseo_anysearch.experiments.tune_runner import TuneRunner

        patches = _patch_ray_tune(fake_with_parameters=fake_with_parameters)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured, "with_parameters was never called"
        return captured[0]

    def test_no_gpu_injects_none(self, tmp_path):
        exp_dict = self._run_and_capture_experiment_dict(False, 1, tmp_path)
        assert exp_dict["training"]["num_gpus"] is None

    def test_single_trial_gpu_injects_one(self, tmp_path):
        exp_dict = self._run_and_capture_experiment_dict(True, 1, tmp_path)
        assert exp_dict["training"]["num_gpus"] == pytest.approx(1.0)

    def test_two_concurrent_gpu_injects_half(self, tmp_path):
        exp_dict = self._run_and_capture_experiment_dict(True, 2, tmp_path)
        assert exp_dict["training"]["num_gpus"] == pytest.approx(0.5)

    def test_output_dir_is_absolute(self, tmp_path):
        """Workers may run with a different CWD; output_dir must be absolute."""
        exp_dict = self._run_and_capture_experiment_dict(False, 1, tmp_path)
        assert Path(exp_dict["experiment"]["output_dir"]).is_absolute()


# ---------------------------------------------------------------------------
# 6. _trial_dirname — short names for Windows MAX_PATH
# ---------------------------------------------------------------------------

class TestTrialDirname:
    """
    _trial_dirname must return a short string that does not embed parameter
    values (which Ray Tune would otherwise embed, exceeding 260 chars).
    """

    def test_returns_trial_prefix_plus_id(self):
        from theseo_anysearch.experiments.tune_runner import _trial_dirname

        mock_trial = MagicMock()
        mock_trial.trial_id = "abc123"
        result = _trial_dirname(mock_trial)
        assert result == "trial_abc123"

    def test_does_not_embed_param_values(self):
        from theseo_anysearch.experiments.tune_runner import _trial_dirname

        mock_trial = MagicMock()
        mock_trial.trial_id = "xyz"
        mock_trial.config = {
            "lr": 0.000123456789,
            "train_batch_size": 8192,
            "kl_coeff": 0.9999,
        }
        result = _trial_dirname(mock_trial)
        assert "0.000123" not in result
        assert "8192" not in result

    def test_length_safe_for_windows_max_path(self, tmp_path):
        """
        Even with a long base path, the trial dir component must be short
        enough that a typical full path stays under 260 chars.
        """
        from theseo_anysearch.experiments.tune_runner import _trial_dirname

        mock_trial = MagicMock()
        mock_trial.trial_id = "a" * 20  # unusually long id
        result = _trial_dirname(mock_trial)
        # The dirname alone should be well under 50 chars
        assert len(result) < 50


# ---------------------------------------------------------------------------
# 7. _detect_num_gpus — num_gpus override
# ---------------------------------------------------------------------------

class TestDetectNumGpusOverride:
    """
    When training.num_gpus is set explicitly, _detect_num_gpus must return
    that value without calling torch.cuda.device_count().
    """

    def test_override_returns_fractional(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        result = _detect_num_gpus(require_gpu=True, num_gpus=0.5)
        assert result == pytest.approx(0.5)

    def test_override_zero_skips_torch(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        # Even with require_gpu=True, num_gpus=0 bypasses the assertion
        with patch("torch.cuda.device_count", return_value=0):
            result = _detect_num_gpus(require_gpu=False, num_gpus=0.0)
        assert result == pytest.approx(0.0)

    def test_override_none_falls_through_to_torch(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        with patch("torch.cuda.device_count", return_value=2):
            result = _detect_num_gpus(require_gpu=False, num_gpus=None)
        assert result == 2

    def test_override_one_does_not_raise_without_cuda(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        # num_gpus override skips require_gpu assertion entirely
        result = _detect_num_gpus(require_gpu=True, num_gpus=1.0)
        assert result == pytest.approx(1.0)

    def test_require_gpu_without_override_raises_if_no_cuda(self):
        from theseo_anysearch.rllib.trainer.base import _detect_num_gpus

        with patch("torch.cuda.device_count", return_value=0):
            with pytest.raises(AssertionError, match="require_gpu"):
                _detect_num_gpus(require_gpu=True, num_gpus=None)


# ---------------------------------------------------------------------------
# 8. _print_sweep_overview — readable sampler descriptions
# ---------------------------------------------------------------------------

class TestPrintSweepOverview:
    """
    The banner must render sampler objects as human-readable descriptions,
    not as repr strings like '<ray.tune.search.sample.Categorical object …>'.
    """

    def _capture_overview(self, search_space: dict) -> str:
        from io import StringIO
        import sys
        from theseo_anysearch.experiments.tune_runner import _print_sweep_overview

        buf = StringIO()
        with patch("sys.stdout", buf):
            _print_sweep_overview(
                config_path=None,
                run_tag="test",
                algorithm="multi_agent_voxel_ppo",
                scheduler="asha",
                num_samples=10,
                max_concurrent=1,
                iterations=100,
                num_env_runners=4,
                require_gpu=False,
                search_space=search_space,
                output_dir=Path("/tmp/out"),
                mlflow_tracking_uri="",
            )
        return buf.getvalue()

    def test_no_object_repr_in_output(self):
        import ray.tune as rt
        search_space = {
            "lr": rt.loguniform(1e-5, 1e-2),
            "batch": rt.choice([256, 512]),
        }
        output = self._capture_overview(search_space)
        assert "object at 0x" not in output, (
            "Sampler repr should not appear in banner; got:\n" + output
        )

    def test_categorical_shows_choices(self):
        import ray.tune as rt
        search_space = {"batch": rt.choice([256, 512, 1024])}
        output = self._capture_overview(search_space)
        assert "256" in output
        assert "512" in output
        assert "1024" in output

    def test_param_name_appears(self):
        import ray.tune as rt
        search_space = {"learning_rate": rt.loguniform(1e-5, 1e-2)}
        output = self._capture_overview(search_space)
        assert "learning_rate" in output

    def test_no_gpu_shows_no(self):
        output = self._capture_overview({})
        assert "GPU           : no" in output

    def test_require_gpu_shows_yes(self):
        from theseo_anysearch.experiments.tune_runner import _print_sweep_overview
        from io import StringIO

        buf = StringIO()
        with patch("sys.stdout", buf):
            _print_sweep_overview(
                config_path=None,
                run_tag="test",
                algorithm="ppo",
                scheduler="asha",
                num_samples=5,
                max_concurrent=1,
                iterations=50,
                num_env_runners=4,
                require_gpu=True,
                search_space={},
                output_dir=Path("/tmp/out"),
                mlflow_tracking_uri="",
            )
        assert "GPU           : yes" in buf.getvalue()
