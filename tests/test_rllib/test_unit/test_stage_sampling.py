from __future__ import annotations

import pytest

from theseo_anysearch.curriculum import StageSamplingContext, StageSamplingStage
from theseo_anysearch.models import (
    WaypointCurriculumConfig,
    WaypointTrainingSamplingConfig,
)
from theseo_anysearch.rllib.trainer.stage_sampling import (
    stage_probabilities,
    stage_sampling,
)
from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
    WaypointCurriculum,
    broadcast_waypoint_curriculum,
)


def context(*success_rates: float | None) -> StageSamplingContext:
    latest = len(success_rates) - 1
    return StageSamplingContext(
        stages=tuple(
            StageSamplingStage(
                index=index,
                start=(index, 0, 0),
                goal=(index + 1, 0, 0),
                age=latest - index,
                is_latest=index == latest,
                evaluation_attempts=10 if rate is not None else 0,
                evaluation_successes=int(rate * 10) if rate is not None else 0,
                evaluation_success_rate=rate,
            )
            for index, rate in enumerate(success_rates)
        )
    )


def test_legacy_sampling_preserves_current_retained_split():
    probabilities = stage_probabilities(
        context(None, None, None),
        WaypointTrainingSamplingConfig(
            current_stage_probability=0.75,
            retained_stage_probability=0.25,
        ),
    )

    assert probabilities == pytest.approx([0.125, 0.125, 0.75])


def test_uniform_sampling_gives_every_stage_equal_probability():
    probabilities = stage_probabilities(
        context(None, None, None),
        WaypointTrainingSamplingConfig(strategy="uniform"),
    )

    assert probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_latest_multiplier_is_normalized_across_all_stages():
    probabilities = stage_probabilities(
        context(None, None, None, None),
        WaypointTrainingSamplingConfig(
            strategy="latest_multiplier",
            latest_multiplier=10.0,
        ),
    )

    assert probabilities == pytest.approx([1 / 13, 1 / 13, 1 / 13, 10 / 13])


def test_recency_sampling_decays_probability_by_stage_age():
    probabilities = stage_probabilities(
        context(None, None, None),
        WaypointTrainingSamplingConfig(
            strategy="recency",
            recency_decay=0.5,
            minimum_weight=0.1,
        ),
    )

    assert probabilities == pytest.approx([1 / 7, 2 / 7, 4 / 7])


def test_inverse_success_prioritizes_poorly_performing_stages():
    probabilities = stage_probabilities(
        context(0.9, 0.5, 0.0),
        WaypointTrainingSamplingConfig(
            strategy="inverse_success",
            minimum_weight=0.1,
        ),
    )

    assert probabilities == pytest.approx([0.0625, 0.3125, 0.625])


def test_custom_sampling_function_uses_function_name_as_strategy():
    @stage_sampling
    def test_custom_latest(context: StageSamplingContext) -> dict[int, float]:
        return {
            stage.index: 5.0 if stage.is_latest else 1.0 for stage in context.stages
        }

    probabilities = stage_probabilities(
        context(None, None, None),
        WaypointTrainingSamplingConfig(strategy="test_custom_latest"),
    )

    assert probabilities == pytest.approx([1 / 7, 1 / 7, 5 / 7])


def test_curriculum_uses_only_latest_stage_evaluation_results():
    curriculum = WaypointCurriculum(
        WaypointCurriculumConfig(
            enabled=True,
            initial_start=(1, 1, 1),
            initial_goal=(2, 2, 2),
            training_sampling={"strategy": "inverse_success"},
        )
    )
    curriculum.advance(1, (2, 2, 2), (4, 4, 4))

    curriculum.record_stage_evaluations([(10, 9), (10, 2)])
    curriculum.record_stage_evaluations([(10, 7), (10, 4)])

    assert curriculum.state.stage_evaluations[0].attempts == 10
    assert curriculum.state.stage_evaluations[0].successes == 7
    assert curriculum.sampling_probabilities() == pytest.approx([1 / 3, 2 / 3])


def test_segmented_route_curriculum_builds_sampling_context_from_route_endpoints():
    curriculum = WaypointCurriculum(
        WaypointCurriculumConfig.model_validate({
            "enabled": True,
            "completion_mode": "continue_route",
            "initial_start": [16, 16, 16],
            "route_length": {"mode": "fixed", "distance": 12},
            "difficulty": {
                "mode": "segment_distance",
                "initial_distance": 2,
                "distance_increment": 1,
                "maximum_distance": 4,
            },
            "training_sampling": {"strategy": "inverse_success"},
        }),
        {
            "grid_size": 32,
            "max_steps": 20,
            "action_mode": "discrete_18",
        },
    )

    assert curriculum.sampling_probabilities() == [1.0]

def test_broadcast_sends_normalized_stage_probabilities_to_environments():
    curriculum = WaypointCurriculum(
        WaypointCurriculumConfig(
            enabled=True,
            initial_start=(1, 1, 1),
            initial_goal=(2, 2, 2),
            training_sampling={
                "strategy": "latest_multiplier",
                "latest_multiplier": 10.0,
            },
        )
    )
    curriculum.advance(1, (2, 2, 2), (4, 4, 4))
    received = []

    class FakeEnvironment:
        def set_waypoint_curriculum(self, stages, probabilities):
            received.append((stages, probabilities))

    class FakeGroup:
        def foreach_env(self, function):
            function(FakeEnvironment())

    class FakeAlgorithm:
        env_runner_group = FakeGroup()

    broadcast_waypoint_curriculum(FakeAlgorithm(), curriculum)

    assert received[0][0] == curriculum.stages()
    assert received[0][1] == pytest.approx([1 / 11, 10 / 11])


@pytest.mark.parametrize(
    "weights, message",
    [
        ({0: 1.0}, "one weight per stage index"),
        ({0: 0.0, 1: 0.0}, "at least one"),
        ({0: -1.0, 1: 1.0}, "non-negative"),
    ],
)
def test_custom_sampling_rejects_invalid_weights(weights, message):
    name = f"test_invalid_{abs(hash(message))}"

    def invalid(_context):
        return weights

    invalid.__name__ = name
    stage_sampling(invalid)

    with pytest.raises(ValueError, match=message):
        stage_probabilities(
            context(None, None),
            WaypointTrainingSamplingConfig(strategy=name),
        )
