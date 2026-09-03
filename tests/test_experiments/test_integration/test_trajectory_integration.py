"""
Integration tests for collect_eval_episode + TrajectoryWriter.

Requires theseo_core wheel with filled_voxels() support.
Run with:
    pytest tests/integration/test_trajectory_integration.py -m integration -v
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _require_filled_voxels():
    try:
        import theseo_core
    except ImportError:
        pytest.skip("theseo_core not installed")
    if not hasattr(theseo_core.PyVoxelEnv, "filled_voxels"):
        pytest.skip("theseo_core wheel missing filled_voxels — rebuild with maturin develop")


# ---------------------------------------------------------------------------
# collect_eval_episode with a trivial algo (always action=0)
# ---------------------------------------------------------------------------

class _VectorNoopAlgo:
    """Minimal algorithm returning the vector_3 center no-op."""

    def compute_single_action(self, obs, policy_id="default_policy", explore=False):
        return [1, 1, 1]

class _ActionZeroAlgo:
    """Minimal duck-type algo that always places (action 0)."""
    def compute_single_action(self, obs, policy_id="default_policy", explore=False):
        return 0


@pytest.fixture
def minimal_env_config():
    """Provide minimal env config."""
    return {
        "agent_count": 1,
        "max_steps": 5,
        "seed": 42,
        "obs_mode": "scalar",
    }


class TestCollectEvalEpisode:
    """Tests CollectEvalEpisode."""
    def setup_method(self):
        _require_filled_voxels()

    def test_returns_voxel_episode_data(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import (
            VoxelEpisodeData, collect_eval_episode,
        )
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        assert isinstance(ep, VoxelEpisodeData)

    def test_steps_list_non_empty(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        assert len(ep.steps) > 0

    def test_steps_count_at_most_max_steps(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        assert len(ep.steps) <= minimal_env_config["max_steps"]

    def test_step_fields_present(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import VoxelStepData, collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        s = ep.steps[0]
        assert isinstance(s, VoxelStepData)
        assert s.step == 0
        assert s.action == 0
        assert isinstance(s.reward, float)
        assert isinstance(s.done, bool)
        assert s.cursor_x >= 1 and s.cursor_y >= 1 and s.cursor_z >= 1

    def test_vector_3_action_is_recorded_as_canonical_noop(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode

        config = {**minimal_env_config, "action_mode": "vector_3", "max_steps": 1}
        episode = collect_eval_episode(_VectorNoopAlgo(), config)
        assert episode.steps[0].action == 26
    def test_total_reward_matches_sum(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        assert abs(ep.total_reward - sum(s.reward for s in ep.steps)) < 1e-5

    def test_init_filled_is_list(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        assert isinstance(ep.init_filled, list)
        for coord in ep.init_filled:
            assert len(coord) == 3

    def test_obs_mode_matches_config(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        assert ep.obs_mode == "scalar"

    def test_max_steps_matches_config(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        assert ep.max_steps == minimal_env_config["max_steps"]

    def test_placed_flag_consistent_with_action(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        for s in ep.steps:
            if s.placed:
                # placed=True only valid when action=Place (0)
                assert s.action == 0

    def test_voxel_count_non_negative(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        ep = collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)
        for s in ep.steps:
            assert s.voxel_count >= 0

    def test_missing_policy_inference_method_raises(self, minimal_env_config):
        """Invalid algorithms must not produce action-zero fallback replays."""
        from theseo_anysearch.experiments.trajectory import collect_eval_episode

        class NoMethodAlgo:
            pass

        with pytest.raises(AttributeError, match="compute_single_action"):
            collect_eval_episode(NoMethodAlgo(), minimal_env_config)


# ---------------------------------------------------------------------------
# TrajectoryWriter end-to-end with real episodes
# ---------------------------------------------------------------------------

class TestTrajectoryWriterWithRealEpisode:
    """Tests TrajectoryWriterWithRealEpisode."""
    def setup_method(self):
        _require_filled_voxels()

    @pytest.fixture
    def real_episode(self, minimal_env_config):
        from theseo_anysearch.experiments.trajectory import collect_eval_episode
        return collect_eval_episode(_ActionZeroAlgo(), minimal_env_config)

    def test_writer_records_and_writes(self, tmp_path, real_episode):
        from theseo_anysearch.experiments.output import OutputStore
        from theseo_anysearch.experiments.trajectory import TrajectoryWriter
        store = OutputStore(tmp_path)
        writer = TrajectoryWriter(store, trajectory_every=1, best_trajectory=True)
        writer.record(real_episode)
        written = writer.on_iteration_end(1, 0.5, "test-exp", "abc12345")
        assert "trajectories/iter_000001.json" in written
        assert "trajectories/best.json" in written

    def test_written_json_has_real_steps(self, tmp_path, real_episode):
        import json
        from theseo_anysearch.experiments.output import OutputStore
        from theseo_anysearch.experiments.trajectory import TrajectoryWriter
        store = OutputStore(tmp_path)
        writer = TrajectoryWriter(store, trajectory_every=1, best_trajectory=False)
        writer.record(real_episode)
        writer.on_iteration_end(1, 0.5, "test-exp", "abc12345")
        data = json.loads(store.read_bytes("trajectories/iter_000001.json").decode())
        assert len(data["episode"]["steps"]) == len(real_episode.steps)
        assert data["episode"]["total_reward"] == pytest.approx(real_episode.total_reward)

    def test_roundtrip_load(self, tmp_path, real_episode):
        import json
        from theseo_anysearch.experiments.output import OutputStore
        from theseo_anysearch.experiments.trajectory import TrajectoryWriter
        store = OutputStore(tmp_path)
        writer = TrajectoryWriter(store, trajectory_every=1, best_trajectory=False)
        writer.record(real_episode)
        writer.on_iteration_end(1, 0.42, "round-trip", "xyz99")
        loaded = TrajectoryWriter.load(store, "trajectories/iter_000001.json")
        assert loaded["experiment_name"] == "round-trip"
        assert loaded["run_id"] == "xyz99"
        assert loaded["episode_reward_mean"] == pytest.approx(0.42)
