"""Tests for reward-component evaluation reporting."""

from theseo_anysearch.experiments.trajectory import VoxelEpisodeData, VoxelStepData
from theseo_anysearch.rllib.trainer.evaluation import EvaluationMetrics


def test_reward_breakdown_is_flattened_for_reporters():
    episode = VoxelEpisodeData(
        agent_count=1,
        max_steps=2,
        obs_mode="scalar",
        init_filled=[],
        steps=[
            VoxelStepData(
                step=0,
                action=13,
                reward=0.19,
                done=True,
                cursor_x=1,
                cursor_y=1,
                cursor_z=2,
                voxel_count=1,
                placed=True,
                reward_breakdown={"step_cost": -0.01, "distance_progress": 0.2},
                termination_reason="success",
            )
        ],
        total_reward=0.19,
        success=True,
        start_pos=(1, 1, 1),
        goal_pos=(1, 1, 2),
        termination_reason="success",
        reward_breakdown={"step_cost": -0.01, "distance_progress": 0.2},
        unshaped_return=-0.01,
    )

    metrics = EvaluationMetrics.from_voxel_episodes(
        [episode],
        {"step_cost": -0.01, "goal_reward": 1.0},
        min_success_rate=1.0,
    ).scalar_metrics()

    assert metrics["evaluation_reward_step_cost_mean"] == -0.01
    assert metrics["evaluation_reward_distance_progress_mean"] == 0.2
    assert metrics["evaluation_unshaped_return_mean"] == -0.01
