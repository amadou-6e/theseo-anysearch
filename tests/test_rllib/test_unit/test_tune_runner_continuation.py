"""Unit tests for tune runner continuation and resume flows."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from ._tune_runner_support import (
    make_experiment_config,
    make_fake_tuner_fit,
    patch_ray_tune,
    write_existing_sweep,
)


class TestTuneContinuation:
    """Verify explicit resume and append behavior for sweeps."""

    def test_existing_sweep_requires_resume_or_extra_trials(self, tmp_path):
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        write_existing_sweep(tmp_path)
        cfg = make_experiment_config(output_dir=str(tmp_path))

        with pytest.raises(FileExistsError, match="resume an interrupted sweep"):
            TuneRunner(cfg, config_path=None, tag="latest").run()

    def test_extra_trials_starts_continuation_segment(self, tmp_path):
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        import ray.tune as ray_tune

        write_existing_sweep(tmp_path)
        captured_kwargs = []
        tune_config_calls = []
        run_config_calls = []

        def fake_with_parameters(fn, **kwargs):
            captured_kwargs.append(kwargs)
            return MagicMock(name="trainable_final")

        def fake_tune_config(**kwargs):
            tune_config_calls.append(kwargs)
            return MagicMock()

        def fake_run_config(**kwargs):
            run_config_calls.append(kwargs)
            return MagicMock()

        cfg = make_experiment_config(output_dir=str(tmp_path))
        patches = [
            patch("ray.init"),
            patch("ray.shutdown"),
            patch.object(ray_tune, "with_resources", MagicMock(return_value=MagicMock())),
            patch.object(ray_tune, "with_parameters", fake_with_parameters),
            patch.object(ray_tune, "TuneConfig", fake_tune_config),
            patch.object(ray_tune, "RunConfig", fake_run_config),
            patch.object(ray_tune, "Tuner", MagicMock(return_value=make_fake_tuner_fit())),
            patch("theseo_anysearch.cli.commands.tune._build_scheduler", return_value=MagicMock()),
            patch("theseo_anysearch.cli.commands.tune._build_search_alg", return_value=None),
            patch("theseo_anysearch.cli.commands.tune._parse_search_space", return_value={}),
            patch("theseo_anysearch.experiments.tune_runner._print_sweep_overview"),
        ]

        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            result = TuneRunner(
                cfg,
                config_path=None,
                tag="latest",
                extra_trials=3,
            ).run()

        assert captured_kwargs
        assert captured_kwargs[0]["trial_prefix"] == "cont_001_"
        assert tune_config_calls[0]["num_samples"] == 3
        assert run_config_calls[0]["name"] == "latest_cont_001"[:24]
        assert result["segment_name"] == "cont_001"

    def test_resume_uses_tuner_restore(self, tmp_path):
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        import ray.tune as ray_tune

        write_existing_sweep(tmp_path)
        cfg = make_experiment_config(output_dir=str(tmp_path))
        restore_calls = []

        def fake_restore(path, **kwargs):
            restore_calls.append({"path": path, **kwargs})
            return make_fake_tuner_fit()

        patches = [
            patch("ray.init"),
            patch("ray.shutdown"),
            patch.object(ray_tune, "with_resources", MagicMock(return_value=MagicMock())),
            patch.object(ray_tune, "with_parameters", MagicMock(return_value=MagicMock())),
            patch.object(ray_tune.Tuner, "can_restore", return_value=True),
            patch.object(ray_tune.Tuner, "restore", side_effect=fake_restore),
            patch("theseo_anysearch.cli.commands.tune._build_scheduler", return_value=MagicMock()),
            patch("theseo_anysearch.cli.commands.tune._build_search_alg", return_value=None),
            patch("theseo_anysearch.cli.commands.tune._parse_search_space", return_value={}),
            patch("theseo_anysearch.experiments.tune_runner._print_sweep_overview"),
        ]

        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            TuneRunner(cfg, config_path=None, tag="latest", resume=True).run()

        assert restore_calls
        assert restore_calls[0]["path"].endswith("ray-store\\latest") or restore_calls[0][
            "path"
        ].endswith("ray-store/latest")

    def test_cpu_only_trials_embed_num_gpus_zero_in_experiment_dict(self, tmp_path):
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        captured_kwargs = []

        def fake_with_parameters(trainable, **kwargs):
            captured_kwargs.append(kwargs)
            return MagicMock()

        cfg = make_experiment_config(
            require_gpu=False,
            num_env_runners=0,
            max_concurrent=4,
            output_dir=str(tmp_path),
        )

        patches = patch_ray_tune(fake_with_parameters=fake_with_parameters)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            TuneRunner(cfg, config_path=None).run()

        assert captured_kwargs
        experiment_dict = captured_kwargs[0]["experiment_dict"]
        assert experiment_dict["training"]["num_gpus"] == pytest.approx(0.0)

    def test_all_error_trials_raise_first_trial_error_context(self, tmp_path):
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        import ray.tune as ray_tune

        fake_results = MagicMock()
        fake_results.get_best_result.side_effect = RuntimeError("No best trial found")
        fake_trial = MagicMock()
        fake_trial.error = "TorchPolicy was not able to find enough GPU IDs! Found [], but num_gpus=1."
        fake_results.__iter__.return_value = iter([fake_trial])

        patches = [
            patch("ray.init"),
            patch("ray.shutdown"),
            patch.object(ray_tune, "with_resources", MagicMock(return_value=MagicMock())),
            patch.object(ray_tune, "with_parameters", MagicMock(return_value=MagicMock())),
            patch.object(ray_tune, "TuneConfig", MagicMock(return_value=MagicMock())),
            patch.object(ray_tune, "RunConfig", MagicMock(return_value=MagicMock())),
            patch.object(
                ray_tune,
                "Tuner",
                MagicMock(return_value=MagicMock(fit=MagicMock(return_value=fake_results))),
            ),
            patch("theseo_anysearch.cli.commands.tune._build_scheduler", return_value=MagicMock()),
            patch("theseo_anysearch.cli.commands.tune._build_search_alg", return_value=None),
            patch("theseo_anysearch.cli.commands.tune._parse_search_space", return_value={}),
            patch("theseo_anysearch.experiments.tune_runner._print_sweep_overview"),
        ]

        cfg = make_experiment_config(output_dir=str(tmp_path))

        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            with pytest.raises(RuntimeError, match="first trial error"):
                TuneRunner(cfg, config_path=None).run()
