"""
Tests for the experiments layer: models, output store, loader, and runner.

ExperimentRunner.run() is tested with a FakeAlgorithm-backed PPOTrainer
so no Ray installation is required.  CLI commands are smoke-tested via
the Typer test runner.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
from theseo_anysearch.experiments.models import (
    CameraPosition,
    ExperimentConfig,
    ExperimentMeta,
    MLflowConfig,
    RendersConfig,
    SweepConfig,
    TuneConfig,
)
from theseo_anysearch.experiments.output import OutputStore
from theseo_anysearch.experiments.runner import ExperimentRunner, InspectResult, RunInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SINGLE_YAML = textwrap.dedent("""\
    experiment:
      name: test-run
      output_dir: {output_dir}
      seed: 7

    env:
      stl_path: /tmp/test.stl
      scale: 1.0
      agent_count: 2
      max_steps: 50
      seed: 0

    training:
      algorithm: ppo
      model: voxel_encoder
      runner: local
      iterations: 3
      checkpoint_interval: 1
      output_dir: {output_dir}
      video_every: 10

    algorithm_config:
      lr: 0.001
      gamma: 0.99
      train_batch_size: 64
      clip_param: 0.2
      num_sgd_iter: 2
      lambda_: 0.95
      kl_coeff: 0.2

    model_config:
      hidden_sizes: [64]
      activation: relu
""")

SWEEP_YAML = textwrap.dedent("""\
    sweep:
      base:
        experiment:
          name: sweep-base
          output_dir: {output_dir}
          seed: 1
        env:
          stl_path: /tmp/test.stl
          scale: 1.0
          agent_count: 2
          max_steps: 50
        training:
          algorithm: ppo
          model: voxel_encoder
          runner: local
          iterations: 2
          checkpoint_interval: 1
          output_dir: {output_dir}
          video_every: 10
        algorithm_config:
          lr: 0.001
          gamma: 0.99
          train_batch_size: 64
        model_config:
          hidden_sizes: [64]
          activation: relu

      experiments:
        - name: sweep-lr-low
          algorithm_config:
            lr: 1.0e-4
        - name: sweep-lr-high
          algorithm_config:
            lr: 1.0e-2
""")


@pytest.fixture()
def single_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "experiment.yaml"
    p.write_text(SINGLE_YAML.format(output_dir=str(tmp_path)))
    return p


@pytest.fixture()
def sweep_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "sweep.yaml"
    p.write_text(SWEEP_YAML.format(output_dir=str(tmp_path)))
    return p


@pytest.fixture()
def experiment_config(single_yaml: Path) -> ExperimentConfig:
    result = load_experiment(single_yaml)
    assert isinstance(result, ExperimentConfig)
    return result


class FakeAlgo:
    """Duck-typed RLlib Algorithm with new-API-stack result keys."""
    def __init__(self) -> None:
        self._step = 0
        self.saved: list[str] = []
        self.restored: list[str] = []

    def train(self) -> dict[str, Any]:
        self._step += 1
        return {
            "env_runners": {
                "episode_return_mean": float(self._step),
                "episode_len_mean": 20.0,
                "num_episodes_lifetime": self._step * 3,
            },
            "training_iteration": self._step,
        }

    def save(self, path: str) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "weights.sentinel").write_text("ok")
        self.saved.append(path)
        return path

    def restore(self, path: str) -> None:
        sentinel = Path(path) / "weights.sentinel"
        if not sentinel.exists():
            raise FileNotFoundError(path)
        self.restored.append(path)


def _make_runner(config: ExperimentConfig) -> ExperimentRunner:
    """Return an ExperimentRunner whose Trainer uses a FakeAlgo."""
    from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

    class _FakePPO(PPOTrainer):
        def _build_algorithm(self) -> FakeAlgo:
            return FakeAlgo()

    original_build = __builtins__  # noqa

    # Monkey-patch _build_trainer inside runner for this test
    import theseo_anysearch.experiments.runner as runner_mod
    original = runner_mod._build_trainer

    def fake_build(cfg, output_dir):
        settings = cfg.to_settings()
        from theseo_anysearch.models import TrainingConfig
        settings = settings.model_copy(
            update={"training": settings.training.model_copy(update={"output_dir": output_dir})}
        )
        return _FakePPO(settings)

    runner_mod._build_trainer = fake_build
    runner = ExperimentRunner(config)
    runner_mod._build_trainer = original  # restore
    # Keep fake_build attached for the duration of the test
    runner._fake_build = fake_build
    # Re-patch at run time by wrapping run()
    _orig_run = runner.run.__func__

    def _patched_run(self_inner):
        runner_mod._build_trainer = self_inner._fake_build
        try:
            return _orig_run(self_inner)
        finally:
            runner_mod._build_trainer = original

    import types
    runner.run = types.MethodType(_patched_run, runner)
    return runner


# ---------------------------------------------------------------------------
# ExperimentConfig / loader tests
# ---------------------------------------------------------------------------

class TestExperimentModels:
    def test_loads_single_yaml(self, single_yaml: Path):
        result = load_experiment(single_yaml)
        assert isinstance(result, ExperimentConfig)

    def test_experiment_name(self, experiment_config: ExperimentConfig):
        assert experiment_config.experiment.name == "test-run"

    def test_algorithm_config_typed(self, experiment_config: ExperimentConfig):
        from theseo_anysearch.rllib.algorithms.models import PPOConfig
        assert isinstance(experiment_config.algorithm_config, PPOConfig)

    def test_model_cfg_typed(self, experiment_config: ExperimentConfig):
        from theseo_anysearch.rllib.models.models import VoxelEncoderConfig
        assert isinstance(experiment_config.model_cfg, VoxelEncoderConfig)

    def test_run_output_dir_includes_name(self, experiment_config: ExperimentConfig):
        assert experiment_config.run_output_dir.name == "test-run"

    def test_renders_default_empty(self, experiment_config: ExperimentConfig):
        assert experiment_config.renders.camera_positions == []

    def test_tune_config_absent(self, experiment_config: ExperimentConfig):
        assert experiment_config.tune_config is None

    def test_loads_sweep_yaml(self, sweep_yaml: Path):
        result = load_experiment(sweep_yaml)
        assert isinstance(result, SweepConfig)

    def test_expand_sweep_produces_two_experiments(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        assert isinstance(sweep, SweepConfig)
        entries = expand_sweep(sweep)
        assert len(entries) == 2

    def test_expand_sweep_names(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        names = [e.experiment.name for e in entries]
        assert "sweep-lr-low" in names
        assert "sweep-lr-high" in names

    def test_expand_sweep_overrides_lr(self, sweep_yaml: Path):
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        lrs = {e.experiment.name: e.algorithm_config.lr for e in entries}
        assert lrs["sweep-lr-low"] == pytest.approx(1e-4)
        assert lrs["sweep-lr-high"] == pytest.approx(1e-2)

    def test_to_settings_returns_settings(self, experiment_config: ExperimentConfig):
        from theseo_anysearch.models import Settings
        s = experiment_config.to_settings()
        assert isinstance(s, Settings)

    def test_anyscale_config_preserved_when_present(self, tmp_path: Path):
        yaml_text = textwrap.dedent(f"""\
            experiment:
              name: anyscale-run
              output_dir: {tmp_path}

            env:
              stl_path: /tmp/test.stl

            training:
              algorithm: ppo
              runner: anyscale

            anyscale:
              cluster_env: env-1
              compute_config: compute-1
              project: proj-1
        """)
        path = tmp_path.joinpath("experiment_anyscale.yaml")
        path.write_text(yaml_text)

        result = load_experiment(path)
        assert isinstance(result, ExperimentConfig)
        assert result.to_settings().anyscale.project == "proj-1"


# ---------------------------------------------------------------------------
# OutputStore tests
# ---------------------------------------------------------------------------

class TestOutputStore:
    def test_write_read_json(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        store.write_json("meta.json", {"key": "value"})
        assert store.read_json("meta.json") == {"key": "value"}

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        store.write_json("nested/deep/file.json", {})
        assert (tmp_path / "run" / "nested" / "deep" / "file.json").exists()

    def test_exists_true(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        store.write_json("x.json", {})
        assert store.exists("x.json")

    def test_exists_false(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        assert not store.exists("missing.json")

    def test_list_returns_relative_paths(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        store.write_json("a/b.json", {})
        store.write_json("c.json", {})
        paths = store.list()
        assert "a/b.json" in paths
        assert "c.json" in paths

    def test_list_dirs(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        (tmp_path / "run" / "checkpoints" / "iter_000001").mkdir(parents=True)
        (tmp_path / "run" / "checkpoints" / "iter_000002").mkdir(parents=True)
        dirs = store.list_dirs("checkpoints")
        assert any("iter_000001" in d for d in dirs)
        assert any("iter_000002" in d for d in dirs)


# ---------------------------------------------------------------------------
# ExperimentRunner.run() tests
# ---------------------------------------------------------------------------

class TestExperimentRunner:
    def _runner_with_fake(self, config: ExperimentConfig) -> ExperimentRunner:
        from theseo_anysearch.rllib.trainer.ppo import PPOTrainer
        import theseo_anysearch.experiments.runner as runner_mod

        class _FakePPO(PPOTrainer):
            def _build_algorithm(self_inner) -> FakeAlgo:
                return FakeAlgo()

        def fake_build(cfg, output_dir):
            settings = cfg.to_settings()
            settings = settings.model_copy(
                update={"training": settings.training.model_copy(update={"output_dir": output_dir})}
            )
            return _FakePPO(settings)

        runner_mod._build_trainer = fake_build
        runner = ExperimentRunner(config)
        # Restore after test via finalizer would be cleaner, but for simplicity
        # restore inline:
        return runner, runner_mod, fake_build

    def test_run_creates_run_json(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

        class _FakePPO(PPOTrainer):
            def _build_algorithm(self) -> FakeAlgo:
                return FakeAlgo()

        orig = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            s = cfg.to_settings()
            s = s.model_copy(update={"training": s.training.model_copy(update={"output_dir": output_dir})})
            return _FakePPO(s)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            info = runner.run()
        finally:
            runner_mod._build_trainer = orig

        assert info.status == "COMPLETED"
        assert len(info.run_id) == 8

    def test_run_output_dir_created(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

        class _FakePPO(PPOTrainer):
            def _build_algorithm(self) -> FakeAlgo:
                return FakeAlgo()

        orig = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            s = cfg.to_settings()
            s = s.model_copy(update={"training": s.training.model_copy(update={"output_dir": output_dir})})
            return _FakePPO(s)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            info = runner.run()
        finally:
            runner_mod._build_trainer = orig

        run_dir = experiment_config.run_output_dir / info.run_id
        assert run_dir.is_dir()

    def test_run_writes_experiment_yaml(self, experiment_config: ExperimentConfig, single_yaml: Path):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

        class _FakePPO(PPOTrainer):
            def _build_algorithm(self) -> FakeAlgo:
                return FakeAlgo()

        orig = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            s = cfg.to_settings()
            s = s.model_copy(update={"training": s.training.model_copy(update={"output_dir": output_dir})})
            return _FakePPO(s)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config, single_yaml)
            info = runner.run()
        finally:
            runner_mod._build_trainer = orig

        run_dir = experiment_config.run_output_dir / info.run_id
        assert (run_dir / "experiment.yaml").exists()

    def test_run_checkpoints_written(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

        class _FakePPO(PPOTrainer):
            def _build_algorithm(self) -> FakeAlgo:
                return FakeAlgo()

        orig = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            s = cfg.to_settings()
            s = s.model_copy(update={"training": s.training.model_copy(update={"output_dir": output_dir})})
            return _FakePPO(s)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            info = runner.run()
        finally:
            runner_mod._build_trainer = orig

        # iterations=3, checkpoint_interval=1 → 3 checkpoints
        assert info.checkpoint_iterations == [1, 2, 3]


# ---------------------------------------------------------------------------
# ExperimentRunner.inspect() tests
# ---------------------------------------------------------------------------

class TestInspect:
    def _create_run(self, tmp_path: Path) -> tuple[Path, str]:
        run_id = "abcd1234"
        run_dir = tmp_path / "test-exp" / run_id
        run_dir.mkdir(parents=True)
        # Minimal run.json
        info = RunInfo(
            run_id=run_id,
            experiment_name="test-exp",
            start_time="2026-01-01T00:00:00+00:00",
            status="COMPLETED",
        )
        (run_dir / "run.json").write_text(info.model_dump_json())
        # Checkpoint dir
        (run_dir / "checkpoints" / "iter_000001").mkdir(parents=True)
        (run_dir / "checkpoints" / "iter_000002").mkdir(parents=True)
        return tmp_path, run_id

    def test_inspect_returns_inspect_result(self, tmp_path: Path):
        base, run_id = self._create_run(tmp_path)
        result = ExperimentRunner.inspect(run_id, base)
        assert isinstance(result, InspectResult)

    def test_inspect_status(self, tmp_path: Path):
        base, run_id = self._create_run(tmp_path)
        result = ExperimentRunner.inspect(run_id, base)
        assert result.status == "COMPLETED"

    def test_inspect_checkpoint_iterations(self, tmp_path: Path):
        base, run_id = self._create_run(tmp_path)
        result = ExperimentRunner.inspect(run_id, base)
        assert result.checkpoint_iterations == [1, 2]

    def test_inspect_unknown_run_raises(self, tmp_path: Path):
        from theseo_anysearch.experiments.runner import _find_run_dir
        with pytest.raises(FileNotFoundError):
            _find_run_dir(tmp_path, "deadbeef")


# ---------------------------------------------------------------------------
# ExperimentRunner.list_runs() tests
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_empty_dir_returns_empty(self, tmp_path: Path):
        runs = ExperimentRunner.list_runs(tmp_path)
        assert runs == []

    def test_finds_runs(self, tmp_path: Path):
        for run_id in ["aabb1122", "ccdd3344"]:
            run_dir = tmp_path / "exp" / run_id
            run_dir.mkdir(parents=True)
            info = RunInfo(
                run_id=run_id,
                experiment_name="exp",
                start_time=f"2026-01-0{run_id[0]}T00:00:00+00:00",
                status="COMPLETED",
            )
            (run_dir / "run.json").write_text(info.model_dump_json())

        runs = ExperimentRunner.list_runs(tmp_path)
        assert len(runs) == 2
        assert {r["run_id"] for r in runs} == {"aabb1122", "ccdd3344"}


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestExperimentCLI:
    def test_experiment_help(self):
        from typer.testing import CliRunner
        from theseo_anysearch.cli.main import app
        result = CliRunner().invoke(app, ["experiment", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_experiment_run_help(self):
        from typer.testing import CliRunner
        from theseo_anysearch.cli.main import app
        result = CliRunner().invoke(app, ["experiment", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_experiment_resume_help(self):
        from typer.testing import CliRunner
        from theseo_anysearch.cli.main import app
        result = CliRunner().invoke(app, ["experiment", "resume", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_inspect_help(self):
        from typer.testing import CliRunner
        from theseo_anysearch.cli.main import app
        result = CliRunner().invoke(app, ["experiment", "inspect", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_list_help(self):
        from typer.testing import CliRunner
        from theseo_anysearch.cli.main import app
        result = CliRunner().invoke(app, ["experiment", "list", "--help"])
        assert result.exit_code == 0

    def test_experiment_repeat_help(self):
        from typer.testing import CliRunner
        from theseo_anysearch.cli.main import app
        result = CliRunner().invoke(app, ["experiment", "repeat", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_root_help_includes_experiment(self):
        from typer.testing import CliRunner
        from theseo_anysearch.cli.main import app
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "experiment" in result.output


# ---------------------------------------------------------------------------
# Shared patch helper
# ---------------------------------------------------------------------------

def _patch_build(runner_mod, config, fake_algo_cls=FakeAlgo):
    """Return (fake_build_fn, orig) for patching runner_mod._build_trainer."""
    from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

    class _FakePPO(PPOTrainer):
        def _build_algorithm(self_inner):
            return fake_algo_cls()

    orig = runner_mod._build_trainer

    def fake_build(cfg, output_dir):
        s = cfg.to_settings()
        s = s.model_copy(
            update={"training": s.training.model_copy(update={"output_dir": output_dir})}
        )
        return _FakePPO(s)

    return fake_build, orig


# ---------------------------------------------------------------------------
# OutputStore — additional coverage
# ---------------------------------------------------------------------------

class TestOutputStoreExtra:
    def test_write_read_bytes(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        data = b"\x00\x01\x02proto"
        store.write_bytes("traj/episode_001.pb", data)
        assert store.read_bytes("traj/episode_001.pb") == data

    def test_write_bytes_creates_parents(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        store.write_bytes("a/b/c.pb", b"x")
        assert (tmp_path / "run" / "a" / "b" / "c.pb").exists()

    def test_write_yaml_copies_file(self, tmp_path: Path):
        src = tmp_path / "config.yaml"
        src.write_text("key: value\n")
        store = OutputStore(tmp_path / "run")
        dest = store.write_yaml("experiment.yaml", src)
        assert dest.read_text() == "key: value\n"

    def test_write_yaml_does_not_modify_source(self, tmp_path: Path):
        src = tmp_path / "config.yaml"
        src.write_text("original: true\n")
        store = OutputStore(tmp_path / "run")
        store.write_yaml("experiment.yaml", src)
        assert src.read_text() == "original: true\n"

    def test_list_empty_prefix_returns_all(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        store.write_json("a.json", {})
        store.write_bytes("b.pb", b"x")
        paths = store.list()
        assert "a.json" in paths
        assert "b.pb" in paths

    def test_list_dirs_empty_when_prefix_missing(self, tmp_path: Path):
        store = OutputStore(tmp_path / "run")
        assert store.list_dirs("checkpoints") == []


# ---------------------------------------------------------------------------
# ExperimentRunner — failure path
# ---------------------------------------------------------------------------

class TestExperimentRunnerFailure:
    def test_failed_run_status_is_failed(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

        class _BrokenAlgo:
            def train(self):
                raise RuntimeError("algo exploded")
            def save(self, path: str) -> str:
                Path(path).mkdir(parents=True, exist_ok=True)
                return path
            def restore(self, path: str) -> None:
                pass

        class _BrokenPPO(PPOTrainer):
            def _build_algorithm(self):
                return _BrokenAlgo()

        orig = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            s = cfg.to_settings()
            s = s.model_copy(
                update={"training": s.training.model_copy(update={"output_dir": output_dir})}
            )
            return _BrokenPPO(s)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            with pytest.raises(RuntimeError, match="algo exploded"):
                runner.run()
        finally:
            runner_mod._build_trainer = orig

        run_dirs = list(experiment_config.run_output_dir.iterdir())
        assert len(run_dirs) == 1
        import json as _json
        info = RunInfo.model_validate(
            _json.loads((run_dirs[0] / "run.json").read_text())
        )
        assert info.status == "FAILED"

    def test_failed_run_end_time_set(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod
        from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

        class _BrokenAlgo:
            def train(self):
                raise ValueError("boom")
            def save(self, path: str) -> str:
                Path(path).mkdir(parents=True, exist_ok=True)
                return path
            def restore(self, path: str) -> None:
                pass

        class _BrokenPPO(PPOTrainer):
            def _build_algorithm(self):
                return _BrokenAlgo()

        orig = runner_mod._build_trainer

        def fake_build(cfg, output_dir):
            s = cfg.to_settings()
            s = s.model_copy(
                update={"training": s.training.model_copy(update={"output_dir": output_dir})}
            )
            return _BrokenPPO(s)

        runner_mod._build_trainer = fake_build
        try:
            runner = ExperimentRunner(experiment_config)
            with pytest.raises(ValueError):
                runner.run()
        finally:
            runner_mod._build_trainer = orig

        import json as _json
        run_dir = next(experiment_config.run_output_dir.iterdir())
        info = RunInfo.model_validate(_json.loads((run_dir / "run.json").read_text()))
        assert info.end_time is not None


# ---------------------------------------------------------------------------
# ExperimentRunner — resume
# ---------------------------------------------------------------------------

class TestExperimentRunnerResume:
    def _do_run(self, config, runner_mod):
        """Complete a fresh run and return its RunInfo."""
        fake_build, orig = _patch_build(runner_mod, config)
        runner_mod._build_trainer = fake_build
        try:
            info = ExperimentRunner(config).run()
        finally:
            runner_mod._build_trainer = orig
        return info

    def test_resume_completes(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        info = self._do_run(experiment_config, runner_mod)

        fake_build, orig = _patch_build(runner_mod, experiment_config)
        runner_mod._build_trainer = fake_build
        try:
            resumed = ExperimentRunner(experiment_config).resume(info.run_id)
        finally:
            runner_mod._build_trainer = orig

        assert resumed.status == "COMPLETED"

    def test_resume_preserves_run_id(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        info = self._do_run(experiment_config, runner_mod)

        fake_build, orig = _patch_build(runner_mod, experiment_config)
        runner_mod._build_trainer = fake_build
        try:
            resumed = ExperimentRunner(experiment_config).resume(info.run_id)
        finally:
            runner_mod._build_trainer = orig

        assert resumed.run_id == info.run_id

    def test_resume_unknown_run_raises(self, experiment_config: ExperimentConfig):
        runner = ExperimentRunner(experiment_config)
        with pytest.raises(FileNotFoundError):
            runner.resume("deadbeef")

    def test_resume_no_checkpoint_raises(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        # Create a run dir with run.json but no checkpoints
        run_id = "nocktst1"
        run_dir = experiment_config.run_output_dir / run_id
        run_dir.mkdir(parents=True)
        info = RunInfo(
            run_id=run_id,
            experiment_name=experiment_config.experiment.name,
            start_time="2026-01-01T00:00:00+00:00",
            status="COMPLETED",
        )
        import json as _json
        (run_dir / "run.json").write_text(_json.dumps(info.model_dump()))

        fake_build, orig = _patch_build(runner_mod, experiment_config)
        runner_mod._build_trainer = fake_build
        try:
            with pytest.raises(FileNotFoundError):
                ExperimentRunner(experiment_config).resume(run_id)
        finally:
            runner_mod._build_trainer = orig


# ---------------------------------------------------------------------------
# ExperimentRunner — repeat
# ---------------------------------------------------------------------------

class TestExperimentRunnerRepeat:
    def _do_run_with_yaml(self, config, yaml_path, runner_mod):
        fake_build, orig = _patch_build(runner_mod, config)
        runner_mod._build_trainer = fake_build
        try:
            info = ExperimentRunner(config, yaml_path).run()
        finally:
            runner_mod._build_trainer = orig
        return info

    def test_repeat_returns_new_run_id(self, experiment_config: ExperimentConfig, single_yaml: Path):
        import theseo_anysearch.experiments.runner as runner_mod

        info1 = self._do_run_with_yaml(experiment_config, single_yaml, runner_mod)

        fake_build, orig = _patch_build(runner_mod, experiment_config)
        runner_mod._build_trainer = fake_build
        try:
            info2 = ExperimentRunner(experiment_config).repeat(info1.run_id)
        finally:
            runner_mod._build_trainer = orig

        assert info2.run_id != info1.run_id

    def test_repeat_status_completed(self, experiment_config: ExperimentConfig, single_yaml: Path):
        import theseo_anysearch.experiments.runner as runner_mod

        info1 = self._do_run_with_yaml(experiment_config, single_yaml, runner_mod)

        fake_build, orig = _patch_build(runner_mod, experiment_config)
        runner_mod._build_trainer = fake_build
        try:
            info2 = ExperimentRunner(experiment_config).repeat(info1.run_id)
        finally:
            runner_mod._build_trainer = orig

        assert info2.status == "COMPLETED"

    def test_repeat_unknown_run_raises(self, experiment_config: ExperimentConfig):
        with pytest.raises(FileNotFoundError):
            ExperimentRunner(experiment_config).repeat("deadbeef")

    def test_repeat_preserves_experiment_name(self, experiment_config: ExperimentConfig):
        import theseo_anysearch.experiments.runner as runner_mod

        fake_build, orig = _patch_build(runner_mod, experiment_config)
        runner_mod._build_trainer = fake_build
        try:
            info1 = ExperimentRunner(experiment_config).run()
            info2 = ExperimentRunner(experiment_config).repeat(info1.run_id)
        finally:
            runner_mod._build_trainer = orig

        assert info2.experiment_name == info1.experiment_name


# ---------------------------------------------------------------------------
# SweepConfig — additional coverage
# ---------------------------------------------------------------------------

class TestSweepConfigExtra:
    def test_base_gamma_preserved_in_each_entry(self, sweep_yaml: Path):
        from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        for entry in entries:
            assert entry.algorithm_config.gamma == pytest.approx(0.99)

    def test_base_env_preserved_in_each_entry(self, sweep_yaml: Path):
        from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        for entry in entries:
            assert entry.env.agent_count == 2

    def test_empty_experiments_list(self, tmp_path: Path):
        from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
        yaml_text = """\
sweep:
  base:
    experiment:
      name: empty-sweep
      output_dir: /tmp/out
      seed: 1
    env:
      stl_path: /tmp/test.stl
      scale: 1.0
      agent_count: 2
      max_steps: 50
    training:
      algorithm: ppo
      model: voxel_encoder
      runner: local
      iterations: 2
      checkpoint_interval: 1
      output_dir: /tmp/out
      video_every: 10
    algorithm_config:
      lr: 0.001
      gamma: 0.99
      train_batch_size: 64
    model_config:
      hidden_sizes: [64]
      activation: relu
  experiments: []
"""
        p = tmp_path / "empty_sweep.yaml"
        p.write_text(yaml_text)
        sweep = load_experiment(p)
        assert expand_sweep(sweep) == []

    def test_each_entry_experiment_name_set(self, sweep_yaml: Path):
        from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
        sweep = load_experiment(sweep_yaml)
        entries = expand_sweep(sweep)
        for entry in entries:
            assert entry.experiment.name != ""
