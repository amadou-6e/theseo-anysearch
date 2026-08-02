"""Regression tests for standardized goal-finding evaluation metrics."""

from __future__ import annotations

import pytest

from theseo_anysearch.rllib.trainer.evaluation.evaluator import EvaluationMetrics
from theseo_anysearch.experiments.trajectory import (
    MultiVoxelEpisodeData,
    MultiVoxelStepData,
    VoxelEpisodeData,
    VoxelStepData,
)


def _episode(
    positions: list[tuple[int, int, int]],
    *,
    success: bool,
    total_reward: float,
) -> VoxelEpisodeData:
    return VoxelEpisodeData(
        agent_count=1,
        max_steps=4,
        obs_mode="box",
        init_filled=[],
        steps=[
            VoxelStepData(
                step=index,
                action=0,
                reward=total_reward / len(positions),
                done=index == len(positions) - 1,
                cursor_x=position[0],
                cursor_y=position[1],
                cursor_z=position[2],
                voxel_count=0,
                placed=False,
            )
            for index, position in enumerate(positions)
        ],
        total_reward=total_reward,
        success=success,
        start_pos=(0, 0, 0),
        goal_pos=(3, 0, 0),
    )


def test_multiple_successes_must_meet_configured_threshold() -> None:
    episodes = [
        _episode([(1, 0, 0), (2, 0, 0), (3, 0, 0)], success=True, total_reward=1.57),
        _episode([(1, 0, 0), (1, 0, 0), (1, 0, 0)], success=False, total_reward=0.37),
        _episode([(0, 1, 0), (0, 1, 0), (0, 1, 0)], success=False, total_reward=-0.23),
    ]

    metrics = EvaluationMetrics.from_voxel_episodes(
        episodes,
        {
            "step_cost": -0.01,
            "collision_cost": 0.0,
            "goal_reward": 1.0,
        },
        min_success_rate=0.5,
    )

    assert metrics.successes == 1
    assert metrics.success_rate == pytest.approx(1 / 3)
    assert metrics.status == "below_success_threshold"
    assert metrics.steps_to_goal == [3]
    assert metrics.steps_to_goal_mean == 3
    assert metrics.terminated_count == 1
    assert metrics.truncated_count == 2
    assert metrics.shaped_return_mean == pytest.approx(0.57)
    assert metrics.unshaped_return_mean == pytest.approx(0.3033333333)


def test_progress_without_success_is_approaching_not_solved() -> None:
    metrics = EvaluationMetrics.from_voxel_episodes(
        [_episode([(1, 0, 0), (2, 0, 0)], success=False, total_reward=0.38)],
        {"step_cost": -0.01, "goal_reward": 1.0},
        min_success_rate=0.5,
    )

    assert metrics.success_rate == 0.0
    assert metrics.goal_progress_mean == 2.0
    assert metrics.minimum_goal_distance_mean == 1.0
    assert metrics.status == "approaching_not_solved"


def test_no_goal_signal_fixture_fails_success_regression() -> None:
    episodes = [
        _episode([(0, 1, 0)] * 4, success=False, total_reward=-0.04)
        for _ in range(4)
    ]

    metrics = EvaluationMetrics.from_voxel_episodes(
        episodes,
        {"step_cost": -0.01, "goal_reward": 1.0},
        min_success_rate=0.5,
    )

    assert metrics.success_rate < 0.5
    assert metrics.status == "not_solved"


def test_success_batch_is_classified_solved() -> None:
    episodes = [
        _episode([(1, 0, 0), (2, 0, 0), (3, 0, 0)], success=True, total_reward=1.57)
        for _ in range(3)
    ]
    episodes.append(
        _episode([(1, 0, 0), (2, 0, 0)], success=False, total_reward=0.38)
    )

    metrics = EvaluationMetrics.from_voxel_episodes(
        episodes,
        {"step_cost": -0.01, "goal_reward": 1.0},
        min_success_rate=0.5,
    )

    assert metrics.success_rate == 0.75
    assert metrics.status == "solved"

def test_multi_agent_batch_uses_same_success_contract() -> None:
    episode = MultiVoxelEpisodeData(
        agent_count=2,
        max_steps=2,
        steps=[
            MultiVoxelStepData(
                step=0,
                actions=[0, 0],
                rewards=[0.1, -0.1],
                done=False,
                cursors=[(1, 0, 0), (0, 1, 0)],
                placed=[False, False],
            ),
            MultiVoxelStepData(
                step=1,
                actions=[0, 0],
                rewards=[1.1, -0.1],
                done=True,
                cursors=[(2, 0, 0), (0, 1, 0)],
                placed=[False, False],
            ),
        ],
        total_rewards=[1.2, -0.2],
        start_positions=[(0, 0, 0), (0, 0, 0)],
        goal_positions=[(2, 0, 0), (2, 0, 0)],
        init_filled=[],
    )

    metrics = EvaluationMetrics.from_multi_voxel_episodes(
        [episode],
        {"step_cost": -0.01, "goal_reward": 1.0},
        min_success_rate=0.5,
    )

    assert metrics.evaluated_episodes == 2
    assert metrics.success_rate == 0.5
    assert metrics.status == "solved"
    assert metrics.terminated_count == 1
    assert metrics.truncated_count == 1
