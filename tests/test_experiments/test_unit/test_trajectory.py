"""
Unit tests for TrajectoryWriter, _build_payload, and collect_eval_episode.

No Rust wheel required — VoxelEpisodeData is constructed directly,
and collect_eval_episode is tested with a fake env + algo.
No Ray required.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from theseo_anysearch.experiments.output import OutputStore
from theseo_anysearch.experiments.trajectory import (
    EpisodeRunMetrics,
    MultiTrajectoryWriter,
    MultiVoxelEpisodeData,
    MultiVoxelStepData,
    TrajectoryWriter,
    VoxelEpisodeData,
    VoxelStepData,
    VoxelMutationData,
    WorldArtifactReference,
    _build_payload,
    _environment_voxel_count,
    _overlay_delta,
    collect_eval_episode,
)
from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(step: int = 0, action: int = 0, reward: float = 0.1,
               placed: bool = True, voxel_count: int = 1) -> VoxelStepData:
    return VoxelStepData(
        step=step, action=action, reward=reward, done=False,
        cursor_x=1, cursor_y=1, cursor_z=1,
        voxel_count=voxel_count, placed=placed,
    )


def _make_episode(total_reward: float = 1.0, n_steps: int = 3) -> VoxelEpisodeData:
    return VoxelEpisodeData(
        agent_count=1,
        max_steps=20,
        obs_mode="scalar",
        init_filled=[],
        steps=[_make_step(i, voxel_count=i + 1) for i in range(n_steps)],
        total_reward=total_reward,
        success=False,
    )


def _make_writer(tmp_path: Path, trajectory_every: int = 5,
                 best_trajectory: bool = True) -> tuple[TrajectoryWriter, OutputStore]:
    store = OutputStore(tmp_path)
    writer = TrajectoryWriter(store, trajectory_every=trajectory_every,
                              best_trajectory=best_trajectory)
    return writer, store


def test_compiled_world_voxel_count_uses_overlay_without_full_enumeration() -> None:
    class RustEnv:
        def overlay_mutations(self):
            return [
                (10, 10, 10, True, 2, False, 0.0),
                (11, 10, 10, True, 1, True, 1.0),
            ]

        def filled_voxels(self):
            raise AssertionError("compiled base must not be enumerated")

    class Env:
        _config = {"compiled_world_path": "pack"}
        _rust_env = RustEnv()

    assert _environment_voxel_count(Env()) == 1


# ---------------------------------------------------------------------------
# No-write cases
# ---------------------------------------------------------------------------

class TestNoWrite:
    """Tests NoWrite."""
    def test_both_disabled_returns_empty(self, tmp_path):
        writer, _ = _make_writer(tmp_path, trajectory_every=0, best_trajectory=False)
        writer.record(_make_episode())
        result = writer.on_iteration_end(5, 0.5, "exp", "run1")
        assert result == []

    def test_empty_buffer_returns_empty(self, tmp_path):
        writer, _ = _make_writer(tmp_path, trajectory_every=5)
        result = writer.on_iteration_end(5, 0.5, "exp", "run1")
        assert result == []

    def test_periodic_off_interval_no_write(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=5, best_trajectory=False)
        writer.record(_make_episode())
        result = writer.on_iteration_end(3, 0.5, "exp", "run1")
        assert result == []
        assert not store.exists("trajectories/iter_000003.json")


# ---------------------------------------------------------------------------
# Periodic save
# ---------------------------------------------------------------------------

class TestPeriodicSave:
    """Tests PeriodicSave."""
    def test_writes_first_iteration_even_off_interval(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=10, best_trajectory=False)
        writer.record(_make_episode())
        result = writer.on_iteration_end(1, 0.5, "exp", "run1")
        assert "trajectories/iter_000001.json" in result
        assert store.exists("trajectories/iter_000001.json")

    def test_writes_at_interval(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=5, best_trajectory=False)
        writer.record(_make_episode())
        result = writer.on_iteration_end(5, 0.5, "exp", "run1")
        assert "trajectories/iter_000005.json" in result
        assert store.exists("trajectories/iter_000005.json")

    def test_writes_at_multiple_intervals(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=5, best_trajectory=False)
        for iteration in [5, 10, 15]:
            writer.record(_make_episode())
            writer.on_iteration_end(iteration, 0.5, "exp", "run1")
        assert store.exists("trajectories/iter_000005.json")
        assert store.exists("trajectories/iter_000010.json")
        assert store.exists("trajectories/iter_000015.json")

    def test_skips_non_interval(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=10, best_trajectory=False)
        for it in [2, 3, 9]:
            writer.record(_make_episode())
            writer.on_iteration_end(it, 0.5, "exp", "run1")
        assert not store.exists("trajectories/iter_000002.json")
        assert not store.exists("trajectories/iter_000009.json")


# ---------------------------------------------------------------------------
# Best-trajectory save
# ---------------------------------------------------------------------------

class TestBestSave:
    """Tests BestSave."""
    def test_writes_best_on_first_improvement(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=0, best_trajectory=True)
        writer.record(_make_episode())
        result = writer.on_iteration_end(1, 0.8, "exp", "run1")
        assert "trajectories/best.json" in result
        assert store.exists("trajectories/best.json")
        assert store.exists("trajectories/best_meta.json")

    def test_best_meta_json_correct(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=0, best_trajectory=True)
        writer.record(_make_episode())
        writer.on_iteration_end(7, 0.9, "exp", "run1")
        meta = store.read_json("trajectories/best_meta.json")
        assert meta["iteration"] == 7
        assert meta["episode_reward_mean"] == pytest.approx(0.9)

    def test_best_updated_on_improvement(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=0, best_trajectory=True)
        writer.record(_make_episode(total_reward=0.5))
        writer.on_iteration_end(1, 0.5, "exp", "run1")
        writer.record(_make_episode(total_reward=1.0))
        writer.on_iteration_end(2, 0.9, "exp", "run1")
        meta = store.read_json("trajectories/best_meta.json")
        assert meta["iteration"] == 2

    def test_best_not_updated_on_regression(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=0, best_trajectory=True)
        writer.record(_make_episode())
        writer.on_iteration_end(1, 0.9, "exp", "run1")
        writer.record(_make_episode())
        result = writer.on_iteration_end(2, 0.3, "exp", "run1")
        assert "trajectories/best.json" not in result
        # meta still points to iteration 1
        meta = store.read_json("trajectories/best_meta.json")
        assert meta["iteration"] == 1

    def test_best_not_updated_on_equal_reward(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=0, best_trajectory=True)
        writer.record(_make_episode())
        writer.on_iteration_end(1, 0.5, "exp", "run1")
        writer.record(_make_episode())
        result = writer.on_iteration_end(2, 0.5, "exp", "run1")
        assert "trajectories/best.json" not in result


# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------

class TestEpisodeSelection:
    """Tests EpisodeSelection."""
    def test_selects_highest_reward_episode(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=1, best_trajectory=False)
        writer.record(_make_episode(total_reward=0.2))
        writer.record(_make_episode(total_reward=1.5))
        writer.record(_make_episode(total_reward=0.7))
        writer.on_iteration_end(1, 0.5, "exp", "run1")
        data = json.loads(store.read_bytes("trajectories/iter_000001.json"))
        assert data["episode"]["total_reward"] == pytest.approx(1.5)

    def test_buffer_cleared_after_on_iteration_end(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=1, best_trajectory=False)
        writer.record(_make_episode())
        writer.on_iteration_end(1, 0.5, "exp", "run1")
        # Second call without recording → empty buffer → no write
        result = writer.on_iteration_end(2, 0.5, "exp", "run1")
        assert result == []


# ---------------------------------------------------------------------------
# JSON payload structure
# ---------------------------------------------------------------------------

class TestPayload:
    """Tests Payload."""
    def test_payload_has_expected_top_level_keys(self):
        ep = _make_episode()
        payload = _build_payload(ep, 10, 0.75, "my-exp", "abc1", init_filled_file="init.npy")
        for key in ("experiment_name", "run_id", "iteration",
                    "episode_reward_mean", "grid_size", "agent_count",
                    "max_steps", "obs_mode", "episode"):
            assert key in payload

    def test_payload_metadata(self):
        ep = _make_episode()
        payload = _build_payload(ep, 10, 0.75, "my-exp", "abc1", init_filled_file="init.npy")
        assert payload["experiment_name"] == "my-exp"
        assert payload["run_id"] == "abc1"
        assert payload["iteration"] == 10
        assert payload["episode_reward_mean"] == pytest.approx(0.75)
        assert payload["grid_size"] == 32

    def test_payload_episode_fields(self):
        ep = _make_episode(total_reward=2.5, n_steps=4)
        payload = _build_payload(ep, 1, 0.5, "e", "r", init_filled_file="init.npy")
        epi = payload["episode"]
        assert epi["total_reward"] == pytest.approx(2.5)
        assert epi["steps_taken"] == 4
        assert len(epi["steps"]) == 4

    def test_payload_step_fields(self):
        ep = _make_episode(n_steps=1)
        payload = _build_payload(ep, 1, 0.5, "e", "r", init_filled_file="init.npy")
        step = payload["episode"]["steps"][0]
        for key in ("step", "action", "reward", "done",
                    "cursor_x", "cursor_y", "cursor_z",
                    "voxel_count", "placed"):
            assert key in step

    def test_payload_init_filled_file_serialised(self):
        ep = _make_episode()
        ep.init_filled = [(1, 2, 3), (4, 5, 6)]
        payload = _build_payload(ep, 1, 0.5, "e", "r", init_filled_file="init.npy")
        assert payload["episode"]["init_filled_file"] == "init.npy"
        assert "init_filled" not in payload["episode"]

    def test_roundtrip_json(self, tmp_path):
        writer, store = _make_writer(tmp_path, trajectory_every=1, best_trajectory=False)
        ep = _make_episode(total_reward=0.77, n_steps=2)
        writer.record(ep)
        writer.on_iteration_end(1, 0.77, "round-trip", "xyz9")
        data = TrajectoryWriter.load(store, "trajectories/iter_000001.json")
        assert data["episode"]["total_reward"] == pytest.approx(0.77)
        assert data["experiment_name"] == "round-trip"
        assert data["run_id"] == "xyz9"
        assert data["episode"]["init_filled_file"].endswith("_init_filled.npy")
        sidecar = Path("trajectories", data["episode"]["init_filled_file"])
        assert store.exists(str(sidecar))

    def test_compiled_world_uses_reference_and_delta_without_base_sidecar(
        self, tmp_path: Path
    ) -> None:
        cache = tmp_path.joinpath("cache")
        compiled = compile_world(
            [BoxSource((1, 1, 1), (2, 2, 2))],
            WorldExtent(x=60_000, y=40_000, z=20_000),
            cache,
        )
        episode = _make_episode(n_steps=1)
        identity = compiled.manifest.identity_sha256
        episode.world = WorldArtifactReference(
            identity_sha256=identity,
            schema_version=compiled.manifest.schema_version,
            coordinate_type=compiled.manifest.coordinate_type,
            extent=compiled.manifest.extent.as_tuple(),
            manifest_path=f"../worlds/{identity}/manifest.json",
            source_root=compiled.root,
        )
        episode.extent = (60_000, 40_000, 20_000)
        episode.steps[0].mutations = [
            VoxelMutationData(
                coordinate=(59_999, 39_999, 19_999),
                occupied=True,
                kind=5,
                active=True,
                reward_weight=1.0,
            )
        ]
        store = OutputStore(tmp_path.joinpath("run"))
        writer = TrajectoryWriter(store, trajectory_every=1, best_trajectory=False)
        writer.record(episode)
        writer.on_iteration_end(1, 1.0, "large", "run")

        payload = store.read_json("trajectories/iter_000001.json")
        assert payload["schema_version"] == 2
        assert payload["world"]["identity_sha256"] == identity
        assert payload["world"]["extent"] == [60_000, 40_000, 20_000]
        assert "init_filled_file" not in payload["episode"]
        assert payload["episode"]["steps"][0]["mutations"][0]["coordinate"] == [
            59_999,
            39_999,
            19_999,
        ]
        assert store.exists(f"worlds/{identity}/manifest.json")
        assert not any(
            path.endswith("_init_filled.npy") for path in store.list("trajectories")
        )

    def test_overlay_delta_encodes_changes_and_disappearances(self) -> None:
        old = {
            (1, 2, 3): VoxelMutationData((1, 2, 3), True, 5, True, 1.0),
            (4, 5, 6): VoxelMutationData((4, 5, 6), True, 5, True, 1.0),
        }
        new = {
            (1, 2, 3): VoxelMutationData((1, 2, 3), False),
            (7, 8, 9): VoxelMutationData((7, 8, 9), True, 1, True, 0.5),
        }

        assert _overlay_delta(old, new) == [
            VoxelMutationData((1, 2, 3), False),
            VoxelMutationData((4, 5, 6), False),
            VoxelMutationData((7, 8, 9), True, 1, True, 0.5),
        ]

    def test_multi_agent_compiled_world_round_trips_overlay_deltas(
        self, tmp_path: Path
    ) -> None:
        compiled = compile_world(
            [BoxSource((1, 1, 1), (1, 1, 1))],
            WorldExtent(x=64, y=48, z=32),
            tmp_path.joinpath("cache"),
        )
        identity = compiled.manifest.identity_sha256
        world = WorldArtifactReference(
            identity_sha256=identity,
            schema_version=1,
            coordinate_type="u32",
            extent=(64, 48, 32),
            manifest_path=f"../worlds/{identity}/manifest.json",
            source_root=compiled.root,
        )
        episode = MultiVoxelEpisodeData(
            agent_count=2,
            max_steps=5,
            steps=[
                MultiVoxelStepData(
                    step=0,
                    actions=[1, 2],
                    rewards=[0.5, 0.25],
                    done=False,
                    cursors=[(2, 2, 2), (3, 3, 3)],
                    placed=[True, False],
                    mutations=[VoxelMutationData((2, 2, 2), True, 5, True, 1.0)],
                )
            ],
            total_rewards=[0.5, 0.25],
            start_positions=[(1, 2, 2), (3, 3, 3)],
            goal_positions=[(4, 2, 2), None],
            init_filled=[],
            extent=(64, 48, 32),
            world=world,
        )
        store = OutputStore(tmp_path.joinpath("run"))
        writer = MultiTrajectoryWriter(
            store, trajectory_every=1, best_trajectory=False
        )
        writer.record(episode)
        writer.on_iteration_end(1, 0.75, "multi", "run")

        payload = store.read_json("trajectories/iter_000001.json")
        assert payload["world"]["identity_sha256"] == identity
        assert payload["episode"]["steps"][0]["mutations"][0]["coordinate"] == [
            2,
            2,
            2,
        ]
        assert "init_filled_file" not in payload["episode"]


# ---------------------------------------------------------------------------
# collect_eval_episode — fake env + algo (no Rust, no Ray)
# ---------------------------------------------------------------------------

def _obs() -> dict:
    """Minimal policy observation for the fake environment."""
    return {"steps_remaining": np.array([1.0], dtype=np.float32)}


class _FakeEnv:
    """
    Simulates a trail-mode VoxelEnv without Rust.

    step_script: list of (action_that_will_be_taken, new_voxel_count, done).
    On each step() call, we pop from the script and return the next obs.
    No _rust_env attribute → collect_eval_episode skips cursor/goal queries.
    """

    def __init__(self, step_script: list[tuple[int, int, bool]]) -> None:
        self._script = list(step_script)
        self._voxel_count = 0

    def reset(self, seed=None):
        self._voxel_count = 0
        return _obs(), {}

    def step(self, action):  # noqa: ARG002
        _action_expected, new_count, done = self._script.pop(0)
        self._voxel_count = new_count
        return _obs(), -0.01, done, False, {}

    def filled_voxel_count(self) -> int:
        return self._voxel_count

    def close(self):
        pass


class _FakeAlgo:
    """Returns a pre-defined sequence of actions."""

    def __init__(self, actions: list[int]) -> None:
        self._actions = list(actions)

    def compute_single_action(self, obs, policy_id="default_policy", explore=False):  # noqa: ARG002
        return self._actions.pop(0)


class TestCollectEvalEpisode:
    """Tests CollectEvalEpisode."""
    def test_trail_movement_placed_true(self):
        """
        In trail mode a movement action (e.g. 2) fills a cell.
        placed should be True whenever voxel_count increases, regardless of action.
        """
        # script: (action, new_voxel_count, done)
        script = [
            (2, 1, False),  # move → trail fill → placed=True
            (2, 2, False),  # move → trail fill → placed=True
            (2, 2, True),   # move → blocked (same count) → placed=False, done
        ]
        algo = _FakeAlgo([2, 2, 2])
        env = _FakeEnv(script)
        ep = collect_eval_episode(algo, {"max_steps": 10}, env=env)

        assert ep.steps[0].placed is True   # voxel_count 0 → 1
        assert ep.steps[1].placed is True   # voxel_count 1 → 2
        assert ep.steps[2].placed is False  # voxel_count unchanged

    def test_explicit_place_action_still_marked(self):
        """action=0 (Place) that fills a cell must still be marked placed=True."""
        script = [(0, 1, True)]
        algo = _FakeAlgo([0])
        env = _FakeEnv(script)
        ep = collect_eval_episode(algo, {"max_steps": 10}, env=env)

        assert ep.steps[0].placed is True

    def test_no_fill_step_placed_false(self):
        """A step that doesn't change voxel_count → placed=False."""
        script = [
            (1, 0, False),   # Remove / no-op, count stays 0
            (0, 0, True),    # Place fails (cell already filled / boundary)
        ]
        algo = _FakeAlgo([1, 0])
        env = _FakeEnv(script)
        ep = collect_eval_episode(algo, {"max_steps": 10}, env=env)

        assert ep.steps[0].placed is False
        assert ep.steps[1].placed is False

    def test_total_placed_count_trail_mode(self):
        """End-to-end: 5 movement actions each fill a cell → 5 placed=True."""
        n = 5
        script = [(2, i + 1, i == n - 1) for i in range(n)]
        algo = _FakeAlgo([2] * n)
        env = _FakeEnv(script)
        ep = collect_eval_episode(algo, {"max_steps": 20}, env=env)

        placed_steps = [s for s in ep.steps if s.placed]
        assert len(placed_steps) == n


class TestEpisodeRunMetrics:
    """Tests EpisodeRunMetrics."""

    def test_from_voxel_episode_counts_collisions(self):
        episode = VoxelEpisodeData(
            agent_count=1,
            max_steps=10,
            obs_mode="radial",
            init_filled=[],
            total_reward=-1.0,
            success=False,
            start_pos=(4, 4, 4),
            goal_pos=(4, 4, 7),
            steps=[
                VoxelStepData(
                    step=0,
                    action=0,
                    reward=-0.55,
                    done=False,
                    cursor_x=4,
                    cursor_y=4,
                    cursor_z=4,
                    voxel_count=1,
                    placed=False,
                ),
                VoxelStepData(
                    step=1,
                    action=1,
                    reward=-0.03,
                    done=False,
                    cursor_x=4,
                    cursor_y=4,
                    cursor_z=5,
                    voxel_count=2,
                    placed=True,
                ),
            ],
        )

        metrics = EpisodeRunMetrics.from_voxel_episode(episode)

        assert metrics.collision_count == 1
        assert metrics.collision_rate == pytest.approx(0.5)
        assert metrics.finish_count == 0
        assert metrics.finish_rate == pytest.approx(0.0)

    def test_from_voxel_episode_computes_success_and_goal_progress(self):
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

        metrics = EpisodeRunMetrics.from_voxel_episode(episode)

        assert metrics.finish_count == 1
        assert metrics.finish_rate == pytest.approx(1.0)
        assert metrics.mean_steps_on_success == pytest.approx(2.0)
        assert metrics.goal_progress_mean == pytest.approx(2.0)

    def test_as_scalar_dict_uses_tensorboard_tags(self):
        metrics = EpisodeRunMetrics(
            collision_count=2,
            collision_rate=0.25,
            finish_count=1,
            finish_rate=1.0,
            mean_steps_on_success=8.0,
            goal_progress_mean=5.0,
        )

        assert metrics.as_scalar_dict() == {
            "eval/task/collision_rate": 0.25,
        }


def test_episode_run_metrics_aggregate_one_evaluation_batch() -> None:
    failed = _make_episode(total_reward=-1.0, n_steps=4)
    successful = _make_episode(total_reward=1.0, n_steps=2)
    successful.success = True

    metrics = EpisodeRunMetrics.from_voxel_episodes([failed, successful])

    assert metrics.finish_count == 1
    assert metrics.finish_rate == pytest.approx(0.5)
    assert metrics.mean_steps_on_success == pytest.approx(2.0)
