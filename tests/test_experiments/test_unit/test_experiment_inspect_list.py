"""Unit tests for experiment inspection and listing."""

from __future__ import annotations

from pathlib import Path

import pytest

from theseo_anysearch.experiments.runner import ExperimentRunner, InspectResult, RunInfo


class TestInspect:
    """Verify inspection of completed experiment runs."""

    def _create_run(self, tmp_path: Path) -> tuple[Path, str]:
        run_id = "abcd1234"
        run_dir = tmp_path.joinpath("test-exp", run_id)
        run_dir.mkdir(parents=True)
        info = RunInfo(
            run_id=run_id,
            experiment_name="test-exp",
            start_time="2026-01-01T00:00:00+00:00",
            status="COMPLETED",
        )
        run_dir.joinpath("run.json").write_text(info.model_dump_json(), encoding="utf-8")
        run_dir.joinpath("checkpoints", "iter_000001").mkdir(parents=True)
        run_dir.joinpath("checkpoints", "iter_000002").mkdir(parents=True)
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


class TestListRuns:
    """Verify discovery of saved experiment runs."""

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        runs = ExperimentRunner.list_runs(tmp_path)
        assert runs == []

    def test_finds_runs(self, tmp_path: Path):
        for run_id in ["aabb1122", "ccdd3344"]:
            run_dir = tmp_path.joinpath("exp", run_id)
            run_dir.mkdir(parents=True)
            info = RunInfo(
                run_id=run_id,
                experiment_name="exp",
                start_time=f"2026-01-0{run_id[0]}T00:00:00+00:00",
                status="COMPLETED",
            )
            run_dir.joinpath("run.json").write_text(info.model_dump_json(), encoding="utf-8")

        runs = ExperimentRunner.list_runs(tmp_path)
        assert len(runs) == 2
        assert {run["run_id"] for run in runs} == {"aabb1122", "ccdd3344"}
