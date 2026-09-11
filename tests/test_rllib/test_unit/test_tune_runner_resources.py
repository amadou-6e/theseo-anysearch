"""Unit tests for tune runner resource construction."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ._tune_runner_support import make_experiment_config, patch_ray_tune


class TestGpuFraction:
    """Verify per-trial GPU allocation math."""

    def _fraction(self, require_gpu: bool, max_concurrent: int) -> float:
        if require_gpu and max_concurrent > 1:
            return 1.0 / max_concurrent
        if require_gpu:
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
        for count in (1, 2, 3, 4, 8):
            fraction = self._fraction(True, count)
            assert fraction * count <= 1.0 + 1e-9

    def test_cpu_only_sweep_forces_explicit_zero_gpu(self):
        from theseo_anysearch.experiments.tune_runner import _tune_trial_num_gpus

        assert _tune_trial_num_gpus(False, 4) == pytest.approx(0.0)

    def test_concurrent_gpu_sweep_splits_one_gpu_evenly(self):
        from theseo_anysearch.experiments.tune_runner import _tune_trial_num_gpus

        assert _tune_trial_num_gpus(True, 8) == pytest.approx(0.125)

    def test_explicit_fraction_overrides_automatic_split(self):
        from theseo_anysearch.experiments.tune_runner import _tune_trial_num_gpus

        assert _tune_trial_num_gpus(True, 8, 0.25) == pytest.approx(0.25)

    def test_cpu_only_sweep_ignores_explicit_fraction(self):
        from theseo_anysearch.experiments.tune_runner import _tune_trial_num_gpus

        assert _tune_trial_num_gpus(False, 1, 0.25) == pytest.approx(0.0)


class TestPlacementGroupBundles:
    """Verify placement group bundle layout math."""

    def _build_bundles(self, num_env_runners: int, gpu_fraction: float) -> list[dict]:
        driver: dict[str, float] = {"CPU": 1.0}
        if gpu_fraction > 0:
            driver["GPU"] = gpu_fraction
        return [driver] + [{"CPU": 1.0}] * num_env_runners

    def test_bundle_count_8_workers(self):
        assert len(self._build_bundles(8, 0.0)) == 9

    def test_bundle_count_12_workers(self):
        assert len(self._build_bundles(12, 0.0)) == 13

    def test_driver_bundle_has_gpu_when_required(self):
        bundles = self._build_bundles(4, 0.5)
        assert bundles[0].get("GPU") == pytest.approx(0.5)

    def test_driver_bundle_has_no_gpu_when_not_required(self):
        bundles = self._build_bundles(4, 0.0)
        assert "GPU" not in bundles[0]

    def test_worker_bundles_have_no_gpu(self):
        bundles = self._build_bundles(4, 1.0)
        for bundle in bundles[1:]:
            assert "GPU" not in bundle

    def test_all_bundles_have_cpu(self):
        bundles = self._build_bundles(4, 0.5)
        for bundle in bundles:
            assert bundle.get("CPU") == pytest.approx(1.0)

    def test_bundle_count_zero_workers(self):
        assert len(self._build_bundles(0, 0.0)) == 1

    @pytest.mark.parametrize("count", [1, 4, 8, 12, 16])
    def test_bundle_count_parametrize(self, count):
        assert len(self._build_bundles(count, 0.0)) == count + 1


class TestPlacementGroupFactoryCreation:
    """Verify real PlacementGroupFactory objects preserve bundle shape."""

    def test_pgf_accepts_12_runner_bundles(self):
        from ray.tune import PlacementGroupFactory

        driver = {"CPU": 1.0}
        workers = [{"CPU": 1.0}] * 12
        pgf = PlacementGroupFactory([driver] + workers, strategy="SPREAD")
        assert len(pgf.bundles) == 13

    def test_pgf_accepts_zero_runner_bundles(self):
        from ray.tune import PlacementGroupFactory

        pgf = PlacementGroupFactory([{"CPU": 1.0}], strategy="SPREAD")
        assert len(pgf.bundles) == 1

    def test_pgf_driver_gpu_fraction_preserved(self):
        from ray.tune import PlacementGroupFactory

        pgf = PlacementGroupFactory(
            [{"CPU": 1.0, "GPU": 0.5}, {"CPU": 1.0}],
            strategy="SPREAD",
        )
        assert pgf.bundles[0].get("GPU") == pytest.approx(0.5)
        assert "GPU" not in pgf.bundles[1]

    def test_tunerunner_passes_pgf_to_with_resources(self, tmp_path: Path):
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        captured_resources = []

        def fake_with_resources(fn, resources):
            captured_resources.append(resources)
            return MagicMock(name="trainable_with_resources")

        cfg = make_experiment_config(
            require_gpu=False,
            num_env_runners=4,
            max_concurrent=1,
            output_dir=str(tmp_path),
        )

        patches = patch_ray_tune(fake_with_resources=fake_with_resources)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured_resources
        resources = captured_resources[0]
        assert isinstance(resources, PlacementGroupFactory)
        assert len(resources.bundles) == 5

    @pytest.mark.parametrize("num_runners", [0, 1, 4, 12])
    def test_tunerunner_pgf_bundle_count_matches_num_env_runners(
        self,
        num_runners: int,
        tmp_path: Path,
    ):
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        captured = []

        def fake_with_resources(fn, resources):
            captured.append(resources)
            return MagicMock()

        cfg = make_experiment_config(
            require_gpu=False,
            num_env_runners=num_runners,
            max_concurrent=1,
            output_dir=str(tmp_path),
        )

        patches = patch_ray_tune(fake_with_resources=fake_with_resources)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured
        pgf = captured[0]
        assert isinstance(pgf, PlacementGroupFactory)
        assert len(pgf.bundles) == 1 + num_runners


    def test_tunerunner_reserves_evaluation_worker_bundles(self, tmp_path: Path):
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        captured = []

        def fake_with_resources(fn, resources):
            captured.append(resources)
            return MagicMock()

        cfg = make_experiment_config(
            require_gpu=False,
            num_env_runners=2,
            evaluation_num_env_runners=3,
            max_concurrent=1,
            output_dir=str(tmp_path),
        )

        patches = patch_ray_tune(fake_with_resources=fake_with_resources)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured
        pgf = captured[0]
        assert isinstance(pgf, PlacementGroupFactory)
        assert len(pgf.bundles) == 1 + 2 + 3

    def test_tunerunner_reserves_rollout_gpu_on_training_workers(self, tmp_path: Path):
        from ray.tune import PlacementGroupFactory
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        captured = []

        def fake_with_resources(fn, resources):
            captured.append(resources)
            return MagicMock()

        cfg = make_experiment_config(
            require_gpu=True,
            num_env_runners=2,
            evaluation_num_env_runners=1,
            max_concurrent=1,
            output_dir=str(tmp_path),
        )
        cfg = cfg.model_copy(update={
            "training": cfg.training.model_copy(update={
                "num_gpus_per_env_runner": 0.25,
            }),
        })

        patches = patch_ray_tune(fake_with_resources=fake_with_resources)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured
        pgf = captured[0]
        assert isinstance(pgf, PlacementGroupFactory)
        assert pgf.bundles[1]["GPU"] == pytest.approx(0.25)
        assert pgf.bundles[2]["GPU"] == pytest.approx(0.25)
        assert "GPU" not in pgf.bundles[3]


class TestWithResourcesOrderingRegression:
    """Verify placement resources wrap the trainable before parameters."""

    def test_with_resources_called_before_with_parameters(self, tmp_path: Path):
        from theseo_anysearch.experiments.tune_runner import TuneRunner, _experiment_trainable

        call_order = []

        def fake_with_resources(fn, resources):
            call_order.append(("with_resources", fn))
            return MagicMock(name="trainable_with_resources")

        def fake_with_parameters(fn, **kwargs):
            call_order.append(("with_parameters", fn))
            return MagicMock(name="trainable_final")

        cfg = make_experiment_config(
            require_gpu=False,
            num_env_runners=2,
            max_concurrent=1,
            output_dir=str(tmp_path),
        )

        patches = patch_ray_tune(fake_with_resources, fake_with_parameters)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                TuneRunner(cfg, config_path=None, tag="test").run()
            except Exception:
                pass

        assert len(call_order) >= 2
        assert call_order[0][0] == "with_resources"
        assert call_order[0][1] is _experiment_trainable
        assert call_order[1][0] == "with_parameters"


class TestReuseActorsFalse:
    """Verify TuneConfig always disables actor reuse."""

    def test_reuse_actors_is_false(self, tmp_path: Path):
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        tune_config_calls: list[dict] = []

        def fake_tune_config(**kwargs):
            tune_config_calls.append(kwargs)
            return MagicMock()

        cfg = make_experiment_config(output_dir=str(tmp_path))

        patches = patch_ray_tune(fake_tune_config=fake_tune_config)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert tune_config_calls
        assert tune_config_calls[0].get("reuse_actors") is False


class TestGpuFractionInjectedIntoExperimentDict:
    """Verify training.num_gpus is injected into the serialized config."""

    def _run_and_capture_experiment_dict(
        self,
        require_gpu: bool,
        max_concurrent: int,
        tmp_path: Path,
    ) -> dict:
        captured = []

        def fake_with_parameters(fn, **kwargs):
            captured.append(kwargs.get("experiment_dict", {}))
            return MagicMock()

        cfg = make_experiment_config(
            require_gpu=require_gpu,
            max_concurrent=max_concurrent,
            output_dir=str(tmp_path),
        )

        from theseo_anysearch.experiments.tune_runner import TuneRunner

        patches = patch_ray_tune(fake_with_parameters=fake_with_parameters)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                TuneRunner(cfg, config_path=None).run()
            except Exception:
                pass

        assert captured
        return captured[0]

    def test_no_gpu_injects_zero(self, tmp_path: Path):
        exp_dict = self._run_and_capture_experiment_dict(False, 1, tmp_path)
        assert exp_dict["training"]["num_gpus"] == pytest.approx(0.0)

    def test_single_trial_gpu_injects_one(self, tmp_path: Path):
        exp_dict = self._run_and_capture_experiment_dict(True, 1, tmp_path)
        assert exp_dict["training"]["num_gpus"] == pytest.approx(1.0)

    def test_two_concurrent_gpu_injects_half(self, tmp_path: Path):
        exp_dict = self._run_and_capture_experiment_dict(True, 2, tmp_path)
        assert exp_dict["training"]["num_gpus"] == pytest.approx(0.5)

    def test_output_dir_is_absolute(self, tmp_path: Path):
        exp_dict = self._run_and_capture_experiment_dict(False, 1, tmp_path)
        assert Path(exp_dict["experiment"]["output_dir"]).is_absolute()
