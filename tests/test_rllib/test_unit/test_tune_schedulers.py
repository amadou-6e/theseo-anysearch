"""
Unit tests for Phase 2 + 3 tune schedulers.

No Ray cluster required — all tests mock ray.tune.schedulers imports and
validate the scheduler / search_alg objects returned by each implementation.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_search_space() -> dict:
    return {"lr": 1e-3, "gamma": 0.99}


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------

class TestTuneRllibModels:
    """Tests TuneRllibModels."""
    def test_random_config_defaults(self):
        from theseo_anysearch.rllib.tune.models import RandomConfig
        cfg = RandomConfig()
        assert cfg.metric == "episode_reward_mean"
        assert cfg.mode == "max"

    def test_hyperband_config_defaults(self):
        from theseo_anysearch.rllib.tune.models import HyperbandConfig
        cfg = HyperbandConfig()
        assert cfg.max_t == 100
        assert cfg.reduction_factor == 3

    def test_hyperband_config_custom(self):
        from theseo_anysearch.rllib.tune.models import HyperbandConfig
        cfg = HyperbandConfig(max_t=50, reduction_factor=4)
        assert cfg.max_t == 50
        assert cfg.reduction_factor == 4

    def test_bohb_config_defaults(self):
        from theseo_anysearch.rllib.tune.models import BOHBConfig
        cfg = BOHBConfig()
        assert cfg.max_t == 100
        assert cfg.reduction_factor == 3

    def test_optuna_config_defaults(self):
        from theseo_anysearch.rllib.tune.models import OptunaConfig
        cfg = OptunaConfig()
        assert cfg.n_startup_trials == 10
        assert cfg.max_t == 100
        assert cfg.grace_period == 10

    def test_cmaes_config_defaults(self):
        from theseo_anysearch.rllib.tune.models import CMAESConfig
        cfg = CMAESConfig()
        assert cfg.sigma0 == pytest.approx(0.5)

    def test_flaml_config_defaults(self):
        from theseo_anysearch.rllib.tune.models import FLAMLConfig
        cfg = FLAMLConfig()
        assert cfg.time_budget_s == 600
        assert cfg.max_iter is None

    def test_flaml_config_custom(self):
        from theseo_anysearch.rllib.tune.models import FLAMLConfig
        cfg = FLAMLConfig(time_budget_s=120, max_iter=50)
        assert cfg.time_budget_s == 120
        assert cfg.max_iter == 50

    def test_asha_config_has_brackets(self):
        from theseo_anysearch.rllib.tune.models import ASHAConfig
        cfg = ASHAConfig()
        assert cfg.brackets == 1

    def test_pbt_config_has_resample_probability(self):
        from theseo_anysearch.rllib.tune.models import PBTConfig
        cfg = PBTConfig()
        assert cfg.resample_probability == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# experiments/models.py TuneConfig
# ---------------------------------------------------------------------------

class TestExperimentsTuneConfig:
    """Tests ExperimentsTuneConfig."""
    def test_default_scheduler_still_asha(self):
        from theseo_anysearch.experiments.models import TuneConfig
        cfg = TuneConfig()
        assert cfg.scheduler == "asha"

    def test_accepts_all_phase2_schedulers(self):
        from theseo_anysearch.experiments.models import TuneConfig
        for name in ("random", "hyperband", "bohb"):
            cfg = TuneConfig(scheduler=name)
            assert cfg.scheduler == name

    def test_accepts_all_phase3_schedulers(self):
        from theseo_anysearch.experiments.models import TuneConfig
        for name in ("optuna", "cmaes", "flaml"):
            cfg = TuneConfig(scheduler=name)
            assert cfg.scheduler == name

    def test_rejects_unknown_scheduler(self):
        from pydantic import ValidationError
        from theseo_anysearch.experiments.models import TuneConfig
        with pytest.raises(ValidationError):
            TuneConfig(scheduler="magic")

    def test_hyperband_config_block_optional(self):
        from theseo_anysearch.experiments.models import HyperbandSchedulerConfig, TuneConfig
        cfg = TuneConfig(
            scheduler="hyperband",
            hyperband_config=HyperbandSchedulerConfig(max_t=30, reduction_factor=4),
        )
        assert cfg.hyperband_config is not None
        assert cfg.hyperband_config.max_t == 30

    def test_bohb_config_block_optional(self):
        from theseo_anysearch.experiments.models import BOHBSchedulerConfig, TuneConfig
        cfg = TuneConfig(
            scheduler="bohb",
            bohb_config=BOHBSchedulerConfig(max_t=80),
        )
        assert cfg.bohb_config.max_t == 80

    def test_optuna_config_block_optional(self):
        from theseo_anysearch.experiments.models import OptunaSchedulerConfig, TuneConfig
        cfg = TuneConfig(
            scheduler="optuna",
            optuna_config=OptunaSchedulerConfig(n_startup_trials=5),
        )
        assert cfg.optuna_config.n_startup_trials == 5

    def test_cmaes_config_block_optional(self):
        from theseo_anysearch.experiments.models import CMAESSchedulerConfig, TuneConfig
        cfg = TuneConfig(
            scheduler="cmaes",
            cmaes_config=CMAESSchedulerConfig(sigma0=0.3),
        )
        assert cfg.cmaes_config.sigma0 == pytest.approx(0.3)

    def test_flaml_config_block_optional(self):
        from theseo_anysearch.experiments.models import FLAMLSchedulerConfig, TuneConfig
        cfg = TuneConfig(
            scheduler="flaml",
            flaml_config=FLAMLSchedulerConfig(time_budget_s=300),
        )
        assert cfg.flaml_config.time_budget_s == 300

    def test_asha_config_block_optional(self):
        from theseo_anysearch.experiments.models import ASHASchedulerConfig, TuneConfig
        cfg = TuneConfig(
            scheduler="asha",
            asha_config=ASHASchedulerConfig(max_t=200, grace_period=20),
        )
        assert cfg.asha_config.max_t == 200


# ---------------------------------------------------------------------------
# BaseTuneConfig ABC — search_alg() default
# ---------------------------------------------------------------------------

class TestBaseTuneConfigSearchAlg:
    """Tests BaseTuneConfigSearchAlg."""
    def test_search_alg_default_returns_none(self):
        from theseo_anysearch.rllib.tune.base import BaseTuneConfig

        class Minimal(BaseTuneConfig):
            def search_space(self): return {}
            def scheduler(self): return object()
            def metric(self): return "x"
            def mode(self): return "max"

        assert Minimal().search_alg() is None


# ---------------------------------------------------------------------------
# RandomSearch
# ---------------------------------------------------------------------------

class TestRandomSearch:
    """Tests RandomSearch."""
    def _make(self):
        from theseo_anysearch.rllib.tune.models import RandomConfig
        from theseo_anysearch.rllib.tune.random import RandomSearch
        return RandomSearch(RandomConfig(), _fake_search_space())

    def test_search_space_passthrough(self):
        rs = self._make()
        assert rs.search_space() == _fake_search_space()

    def test_metric_and_mode(self):
        rs = self._make()
        assert rs.metric() == "episode_reward_mean"
        assert rs.mode() == "max"

    def test_search_alg_is_none(self):
        rs = self._make()
        assert rs.search_alg() is None

    def test_scheduler_is_fifo(self):
        mock_fifo = MagicMock()
        # Patch at source so the deferred `from ray.tune.schedulers import FIFOScheduler` sees it
        with patch("ray.tune.schedulers.FIFOScheduler", return_value=mock_fifo) as MockFIFO:
            rs = self._make()
            sched = rs.scheduler()
        MockFIFO.assert_called_once()
        assert sched is mock_fifo


# ---------------------------------------------------------------------------
# HyperbandSearch
# ---------------------------------------------------------------------------

class TestHyperbandSearch:
    """Tests HyperbandSearch."""
    def _make(self, max_t=50, reduction_factor=3):
        from theseo_anysearch.rllib.tune.hyperband import HyperbandSearch
        from theseo_anysearch.rllib.tune.models import HyperbandConfig
        return HyperbandSearch(
            HyperbandConfig(max_t=max_t, reduction_factor=reduction_factor),
            _fake_search_space(),
        )

    def test_search_alg_is_none(self):
        hs = self._make()
        assert hs.search_alg() is None

    def test_scheduler_called_with_correct_args(self):
        mock_sched = MagicMock()
        with patch("ray.tune.schedulers.HyperBandScheduler", return_value=mock_sched) as MockHB:
            hs = self._make(max_t=75, reduction_factor=4)
            sched = hs.scheduler()
        MockHB.assert_called_once_with(
            time_attr="training_iteration",
            max_t=75,
            reduction_factor=4,
        )
        assert sched is mock_sched

    def test_metric_mode(self):
        hs = self._make()
        assert hs.metric() == "episode_reward_mean"
        assert hs.mode() == "max"


# ---------------------------------------------------------------------------
# BOHBSearch
# ---------------------------------------------------------------------------

class TestBOHBSearch:
    """Tests BOHBSearch."""
    def _make(self):
        from theseo_anysearch.rllib.tune.bohb import BOHBSearch
        from theseo_anysearch.rllib.tune.models import BOHBConfig
        return BOHBSearch(BOHBConfig(max_t=40, reduction_factor=3), _fake_search_space())

    def test_scheduler_uses_hyperband_for_bohb(self):
        mock_sched = MagicMock()
        mock_module = MagicMock()
        mock_module.HyperBandForBOHB = MagicMock(return_value=mock_sched)
        import sys
        with patch.dict(sys.modules, {"ray.tune.schedulers.hb_bohb": mock_module}):
            b = self._make()
            sched = b.scheduler()
        mock_module.HyperBandForBOHB.assert_called_once_with(
            time_attr="training_iteration", max_t=40, reduction_factor=3
        )
        assert sched is mock_sched

    def test_search_alg_uses_tune_bohb(self):
        mock_alg = MagicMock()
        mock_module = MagicMock()
        mock_module.TuneBOHB = MagicMock(return_value=mock_alg)
        import sys
        with patch.dict(sys.modules, {"ray.tune.search.bohb": mock_module}):
            b = self._make()
            alg = b.search_alg()
        mock_module.TuneBOHB.assert_called_once_with(metric="episode_reward_mean", mode="max")
        assert alg is mock_alg

    def test_missing_configspace_raises_import_error(self):
        import sys
        b = self._make()
        with patch.dict(sys.modules, {"ray.tune.schedulers.hb_bohb": None}):
            with pytest.raises(ImportError, match="ConfigSpace"):
                b.scheduler()


# ---------------------------------------------------------------------------
# OptunaSearch
# ---------------------------------------------------------------------------

class TestOptunaSearch:
    """Tests OptunaSearch."""
    def _make(self):
        from theseo_anysearch.rllib.tune.models import OptunaConfig
        from theseo_anysearch.rllib.tune.optuna import OptunaSearch
        return OptunaSearch(OptunaConfig(n_startup_trials=7), _fake_search_space())

    def test_scheduler_is_asha(self):
        mock_sched = MagicMock()
        with patch("ray.tune.schedulers.ASHAScheduler", return_value=mock_sched) as MockASHA:
            o = self._make()
            sched = o.scheduler()
        MockASHA.assert_called_once()
        assert sched is mock_sched

    def test_search_alg_calls_optuna(self):
        mock_alg = MagicMock()
        mock_module = MagicMock()
        mock_module.OptunaSearch = MagicMock(return_value=mock_alg)
        import sys
        with patch.dict(sys.modules, {"ray.tune.search.optuna": mock_module}):
            o = self._make()
            alg = o.search_alg()
        mock_module.OptunaSearch.assert_called_once_with(
            metric="episode_reward_mean", mode="max", n_startup_trials=7
        )
        assert alg is mock_alg

    def test_missing_optuna_raises_import_error(self):
        import sys
        o = self._make()
        with patch.dict(sys.modules, {"ray.tune.search.optuna": None}):
            with pytest.raises(ImportError, match="optuna"):
                o.search_alg()


# ---------------------------------------------------------------------------
# CMAESSearch
# ---------------------------------------------------------------------------

class TestCMAESSearch:
    """Tests CMAESSearch."""
    def _make(self, sigma0=0.5):
        from theseo_anysearch.rllib.tune.cmaes import CMAESSearch
        from theseo_anysearch.rllib.tune.models import CMAESConfig
        return CMAESSearch(CMAESConfig(sigma0=sigma0), _fake_search_space())

    def test_scheduler_is_fifo(self):
        mock_fifo = MagicMock()
        with patch("ray.tune.schedulers.FIFOScheduler", return_value=mock_fifo):
            c = self._make()
            sched = c.scheduler()
        assert sched is mock_fifo

    def test_missing_nevergrad_raises_import_error(self):
        import sys
        c = self._make()
        with patch.dict(sys.modules, {"nevergrad": None, "ray.tune.search.nevergrad": None}):
            with pytest.raises(ImportError, match="nevergrad"):
                c.search_alg()


# ---------------------------------------------------------------------------
# FLAMLSearch
# ---------------------------------------------------------------------------

class TestFLAMLSearch:
    """Tests FLAMLSearch."""
    def _make(self, time_budget_s=300):
        from theseo_anysearch.rllib.tune.flaml import FLAMLSearch
        from theseo_anysearch.rllib.tune.models import FLAMLConfig
        return FLAMLSearch(FLAMLConfig(time_budget_s=time_budget_s), _fake_search_space())

    def test_scheduler_is_fifo(self):
        mock_fifo = MagicMock()
        with patch("ray.tune.schedulers.FIFOScheduler", return_value=mock_fifo):
            f = self._make()
            sched = f.scheduler()
        assert sched is mock_fifo

    def test_search_alg_calls_cfo(self):
        mock_alg = MagicMock()
        mock_module = MagicMock()
        mock_module.CFO = MagicMock(return_value=mock_alg)
        import sys
        with patch.dict(sys.modules, {"ray.tune.search.flaml": mock_module}):
            f = self._make(time_budget_s=300)
            alg = f.search_alg()
        mock_module.CFO.assert_called_once_with(
            metric="episode_reward_mean", mode="max", time_budget_s=300
        )
        assert alg is mock_alg

    def test_missing_flaml_raises_import_error(self):
        import sys
        f = self._make()
        with patch.dict(sys.modules, {"ray.tune.search.flaml": None}):
            with pytest.raises(ImportError, match="flaml"):
                f.search_alg()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestTuneRegistry:
    """Tests TuneRegistry."""
    def test_supported_schedulers_contains_all_phase2_phase3(self):
        from theseo_anysearch.rllib.tune import supported_schedulers
        names = supported_schedulers()
        for name in ("random", "hyperband", "bohb", "optuna", "cmaes", "flaml"):
            assert name in names

    def test_get_tune_impl_random(self):
        from theseo_anysearch.rllib.tune import get_tune_impl
        from theseo_anysearch.rllib.tune.models import RandomConfig
        from theseo_anysearch.rllib.tune.random import RandomSearch
        impl = get_tune_impl("random", RandomConfig(), _fake_search_space())
        assert isinstance(impl, RandomSearch)

    def test_get_tune_impl_hyperband(self):
        from theseo_anysearch.rllib.tune import get_tune_impl
        from theseo_anysearch.rllib.tune.hyperband import HyperbandSearch
        from theseo_anysearch.rllib.tune.models import HyperbandConfig
        impl = get_tune_impl("hyperband", HyperbandConfig(), _fake_search_space())
        assert isinstance(impl, HyperbandSearch)

    def test_get_tune_impl_unknown_raises(self):
        from theseo_anysearch.rllib.tune import get_tune_impl
        from theseo_anysearch.rllib.tune.models import RandomConfig
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_tune_impl("magic", RandomConfig(), {})

    def test_get_tune_impl_case_insensitive(self):
        from theseo_anysearch.rllib.tune import get_tune_impl
        from theseo_anysearch.rllib.tune.models import RandomConfig
        impl = get_tune_impl("RANDOM", RandomConfig(), {})
        assert impl is not None


# ---------------------------------------------------------------------------
# CLI _build_scheduler
# ---------------------------------------------------------------------------

class TestCliBuildScheduler:
    """Tests CliBuildScheduler."""
    def _call(self, name, max_iter=10):
        from theseo_anysearch.cli.commands.tune import _build_scheduler
        return _build_scheduler(
            scheduler_name=name,
            max_iterations=max_iter,
            hyperparam_mutations={},
        )

    def test_random_returns_fifo(self):
        mock_fifo = MagicMock()
        with patch("ray.tune.schedulers.FIFOScheduler", return_value=mock_fifo):
            sched = self._call("random")
        assert sched is mock_fifo

    def test_hyperband_builds_correctly(self):
        mock_sched = MagicMock()
        with patch("ray.tune.schedulers.HyperBandScheduler", return_value=mock_sched) as MockHB:
            sched = self._call("hyperband", max_iter=30)
        MockHB.assert_called_once_with(
            time_attr="training_iteration", max_t=30, reduction_factor=3
        )
        assert sched is mock_sched

    def test_cmaes_returns_fifo(self):
        mock_fifo = MagicMock()
        with patch("ray.tune.schedulers.FIFOScheduler", return_value=mock_fifo):
            sched = self._call("cmaes")
        assert sched is mock_fifo

    def test_flaml_returns_fifo(self):
        mock_fifo = MagicMock()
        with patch("ray.tune.schedulers.FIFOScheduler", return_value=mock_fifo):
            sched = self._call("flaml")
        assert sched is mock_fifo

    def test_optuna_returns_asha(self):
        mock_asha = MagicMock()
        with patch("ray.tune.schedulers.ASHAScheduler", return_value=mock_asha):
            sched = self._call("optuna", max_iter=20)
        assert sched is mock_asha

    def test_unknown_raises_value_error(self):
        from theseo_anysearch.cli.commands.tune import _build_scheduler
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            _build_scheduler("magic", 10, {})

    def test_all_schedulers_constant_complete(self):
        from theseo_anysearch.cli.commands.tune import _ALL_SCHEDULERS
        for name in ("asha", "pbt", "random", "hyperband", "bohb", "optuna", "cmaes", "flaml"):
            assert name in _ALL_SCHEDULERS


# ---------------------------------------------------------------------------
# CLI _build_search_alg
# ---------------------------------------------------------------------------

class TestCliBuildSearchAlg:
    """Tests CliBuildSearchAlg."""
    def _call(self, name, extra_modules: dict | None = None):
        import sys
        from theseo_anysearch.cli.commands.tune import _build_search_alg
        modules = extra_modules or {}
        with patch.dict(sys.modules, modules):
            return _build_search_alg(
                scheduler_name=name, metric="episode_reward_mean", mode="max"
            )

    def test_asha_returns_none(self):
        assert self._call("asha") is None

    def test_pbt_returns_none(self):
        assert self._call("pbt") is None

    def test_random_returns_none(self):
        assert self._call("random") is None

    def test_hyperband_returns_none(self):
        assert self._call("hyperband") is None

    def test_bohb_returns_alg(self):
        mock_alg = MagicMock()
        mock_module = MagicMock()
        mock_module.TuneBOHB = MagicMock(return_value=mock_alg)
        result = self._call("bohb", extra_modules={"ray.tune.search.bohb": mock_module})
        assert result is mock_alg

    def test_optuna_returns_alg(self):
        mock_alg = MagicMock()
        mock_module = MagicMock()
        mock_module.OptunaSearch = MagicMock(return_value=mock_alg)
        result = self._call("optuna", extra_modules={"ray.tune.search.optuna": mock_module})
        assert result is mock_alg

    def test_flaml_returns_alg(self):
        mock_alg = MagicMock()
        mock_module = MagicMock()
        mock_module.CFO = MagicMock(return_value=mock_alg)
        result = self._call("flaml", extra_modules={"ray.tune.search.flaml": mock_module})
        assert result is mock_alg

    def test_cmaes_missing_nevergrad_raises(self):
        import sys
        from theseo_anysearch.cli.commands.tune import _build_search_alg
        with patch.dict(sys.modules, {"nevergrad": None, "ray.tune.search.nevergrad": None}):
            with pytest.raises(ValueError, match="nevergrad"):
                _build_search_alg("cmaes", "x", "max")


# ---------------------------------------------------------------------------
# _real_trainable: must call ray.tune.report, NOT ray.train.report
# ---------------------------------------------------------------------------
#
# Regression: the original code called `_train.report(...)` (ray.train.report).
# In Ray 2.54, calling ray.train.report inside a Tune trainable raises
# DeprecationWarning which surfaces as an error and kills every trial.
# Fix: use `_tune.report(...)` (ray.tune.report) instead.

class TestRealTrainableReport:
    """Tests RealTrainableReport."""
    def _minimal_config(self) -> dict:
        return {
            "lr": 3e-4, "gamma": 0.99, "train_batch_size": 512,
            "clip_param": 0.2, "kl_coeff": 0.2, "num_sgd_iter": 5,
            "lambda_": 0.95, "num_layers": 2, "layer_size": 64,
        }

    def _run(self, tmp_path, fake_results):
        """
        Call _real_trainable with a mock PPOTrainer and a mock ray.tune,
        then return the mock tune object so callers can assert on .report.
        """
        import ray.tune as _real_ray_tune

        mock_tune = MagicMock(spec=_real_ray_tune)
        mock_tune.get_context.return_value = MagicMock(
            get_trial_id=MagicMock(return_value="t0")
        )

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = fake_results
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)

        (tmp_path / "geo.stl").write_bytes(b"")

        with patch("ray.tune", mock_tune), \
             patch("theseo_anysearch.rllib.algorithms.ppo.PPOTrainer", mock_trainer_cls):
            from theseo_anysearch.cli.commands.tune import _real_trainable
            _real_trainable(
                config=self._minimal_config(),
                stl=str(tmp_path / "geo.stl"),
                agents=2,
                max_steps=50,
                seed=0,
                max_iterations=1,
                output_dir=str(tmp_path / "out"),
                metric="episode_reward_mean",
                mlflow_tracking_uri="",
                mlflow_experiment_name="test",
                mlflow_parent_run_id="",
            )

        return mock_tune

    def _make_result(self, reward=1.0):
        from theseo_anysearch.rllib.trainer.results import TrainResult
        return TrainResult(
            iteration=1,
            episode_reward_mean=reward,
            episode_len_mean=10.0,
            episodes_total=5,
            elapsed_s=0.1,
        )

    def test_tune_report_called_once_per_result(self, tmp_path):
        mock_tune = self._run(tmp_path, [self._make_result()])
        mock_tune.report.assert_called_once()

    def test_tune_report_called_for_each_result(self, tmp_path):
        fake = [self._make_result(r) for r in (1.0, 2.0, 3.0)]
        mock_tune = self._run(tmp_path, fake)
        assert mock_tune.report.call_count == 3

    def test_report_payload_contains_metric(self, tmp_path):
        mock_tune = self._run(tmp_path, [self._make_result(reward=7.5)])
        call_args = mock_tune.report.call_args[0][0]
        assert call_args["episode_reward_mean"] == pytest.approx(7.5)

    def test_report_payload_contains_standard_keys(self, tmp_path):
        mock_tune = self._run(tmp_path, [self._make_result()])
        call_args = mock_tune.report.call_args[0][0]
        for key in ("episode_reward_mean", "episode_len_mean",
                    "episodes_total", "training_iteration"):
            assert key in call_args, f"missing key '{key}' in report payload"

    def test_train_report_never_called(self, tmp_path):
        """ray.train.report must not be called — that API raises in Ray 2.54 Tune."""
        import ray.train as _real_ray_train
        mock_train = MagicMock(spec=_real_ray_train)

        with patch("ray.train", mock_train):
            self._run(tmp_path, [self._make_result()])

        mock_train.report.assert_not_called()

    def test_fractional_gpu_is_forwarded_to_trainer(self, tmp_path):
        import ray.tune as _real_ray_tune

        mock_tune = MagicMock(spec=_real_ray_tune)
        mock_tune.get_context.return_value = MagicMock(
            get_trial_id=MagicMock(return_value="t0")
        )
        mock_trainer_cls = MagicMock()
        mock_trainer_cls.return_value.train.return_value = []
        (tmp_path / "geo.stl").write_bytes(b"")

        with patch("ray.tune", mock_tune), \
             patch("theseo_anysearch.rllib.algorithms.ppo.PPOTrainer", mock_trainer_cls):
            from theseo_anysearch.cli.commands.tune import _real_trainable
            _real_trainable(
                config=self._minimal_config(),
                stl=str(tmp_path / "geo.stl"),
                agents=1,
                max_steps=50,
                seed=0,
                max_iterations=1,
                output_dir=str(tmp_path / "out"),
                metric="episode_reward_mean",
                num_gpus=0.25,
            )

        settings = mock_trainer_cls.call_args.args[0]
        assert settings.training.require_gpu is True
        assert settings.training.num_gpus == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _make_trainable: paths passed to Ray workers must be absolute
# ---------------------------------------------------------------------------
#
# Regression: relative paths break when a Ray worker process starts from a
# different CWD.  _make_trainable must call .resolve() on both stl and
# output_dir before passing them to tune.with_parameters.

class TestMakeTrainablePaths:
    """Tests MakeTrainablePaths."""
    def _call(self, tmp_path):
        """Return the kwargs captured by tune.with_parameters."""
        mock_tune = MagicMock()
        captured = {}

        def _fake_with_parameters(fn, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        mock_tune.with_parameters = _fake_with_parameters

        stl = tmp_path / "geo.stl"
        out = tmp_path / "out"

        with patch("ray.tune", mock_tune), \
             patch("theseo_anysearch.cli.commands.tune.tune", mock_tune):
            from theseo_anysearch.cli.commands.tune import _make_trainable
            _make_trainable(
                stl=stl,
                agents=2,
                max_steps=100,
                seed=0,
                max_iterations=5,
                output_dir=out,
                metric="episode_reward_mean",
                mlflow_tracking_uri="",
                mlflow_experiment_name="test",
                mlflow_parent_run_id="",
            )

        return captured

    def test_stl_path_is_absolute(self, tmp_path):
        captured = self._call(tmp_path)
        from pathlib import Path
        assert Path(captured["stl"]).is_absolute()

    def test_output_dir_is_absolute(self, tmp_path):
        captured = self._call(tmp_path)
        from pathlib import Path
        assert Path(captured["output_dir"]).is_absolute()

    def test_stl_is_string_not_path_object(self, tmp_path):
        captured = self._call(tmp_path)
        assert isinstance(captured["stl"], str)

    def test_output_dir_is_string_not_path_object(self, tmp_path):
        captured = self._call(tmp_path)
        assert isinstance(captured["output_dir"], str)

    def test_fractional_gpu_resource_is_preserved(self, tmp_path):
        mock_tune = MagicMock()
        parameterized = MagicMock()
        mock_tune.with_parameters.return_value = parameterized

        with patch("ray.tune", mock_tune), \
             patch("theseo_anysearch.cli.commands.tune.tune", mock_tune):
            from theseo_anysearch.cli.commands.tune import _make_trainable
            _make_trainable(
                stl=tmp_path / "geo.stl",
                agents=1,
                max_steps=50,
                seed=0,
                max_iterations=1,
                output_dir=tmp_path / "out",
                metric="episode_reward_mean",
                num_gpus=0.25,
            )

        mock_tune.with_resources.assert_called_once_with(
            parameterized, resources={"gpu": 0.25}
        )


# ---------------------------------------------------------------------------
# storage_path: must use os.path.normpath to avoid mixed separators
# ---------------------------------------------------------------------------
#
# Regression: on Windows, Ray internally uses Path.as_posix() which produces
# forward slashes (C:/foo/bar), then os.path.join appends with backslash,
# creating mixed paths like C:/foo/bar\\.tmpfile that the OS cannot resolve.
# [Errno 2] No such file or directory: '...driver_artifacts\\.xxx-tmp-xxx.json'
# Fix: storage_path = os.path.normpath(str(output_dir.resolve()))

class TestStoragePathNormalization:
    """Tests StoragePathNormalization."""
    def test_normpath_eliminates_mixed_separators(self):
        """A forward-slash Windows path becomes all-native-separator after normpath."""
        import os
        mixed = "C:/Users/foo/runtime/tune/ppo_arch_search"
        normalised = os.path.normpath(mixed)
        # On Windows the separator is backslash; on POSIX normpath is a no-op.
        assert "/" not in normalised or os.sep == "/"

    def test_normpath_is_idempotent(self):
        """Applying normpath twice gives the same result."""
        import os
        from pathlib import Path
        p = Path("runtime/tune/ppo_arch_search")
        first = os.path.normpath(str(p.resolve()))
        second = os.path.normpath(first)
        assert first == second

    def test_normpath_produces_no_double_separators(self):
        """The normalised path must not contain consecutive separators."""
        import os
        from pathlib import Path
        p = Path("runtime") / "tune" / "experiment"
        result = os.path.normpath(str(p.resolve()))
        assert os.sep * 2 not in result
