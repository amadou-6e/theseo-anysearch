"""Backend smoke tests for mocked policy explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from theseo_anysearch.rllib.explain import (
    ExplanationRequest,
    FeatureSchema,
    LinearMockPolicyScorer,
    ObservationTrace,
    ObservationTraceStep,
    explain_policy,
)


ACTION_PLUS_X = 21
ACTION_PLUS_Z = 13
BOUNDARY_RAY_TYPE = 4.0 / 5.0


def collision_observation() -> dict[str, np.ndarray]:
    """Build a radial observation with a visible +X boundary collision."""

    ray_hits = np.zeros(26, dtype=np.float32)
    ray_hit_types = np.zeros(26, dtype=np.float32)
    ray_hits[ACTION_PLUS_X] = 1.0
    ray_hit_types[ACTION_PLUS_X] = BOUNDARY_RAY_TYPE
    return {
        "steps_remaining": np.asarray([0.95], dtype=np.float32),
        "goal_distance": np.asarray([0.20], dtype=np.float32),
        "goal_direction": np.asarray([0.20, 0.00, 0.05], dtype=np.float32),
        "cursor_pos": np.asarray([0.10, 0.10, 0.10], dtype=np.float32),
        "ray_hits": ray_hits,
        "ray_hit_types": ray_hit_types,
    }


def trace_with_visible_collision(observation: dict[str, np.ndarray]) -> ObservationTrace:
    """Build a two-step trace with one collision and one background step."""

    background = {
        name: value.copy()
        for name, value in observation.items()
    }
    background["ray_hits"][ACTION_PLUS_X] = 0.0
    background["ray_hit_types"][ACTION_PLUS_X] = 0.0
    return ObservationTrace(
        [
            ObservationTraceStep(
                step=0,
                observation=observation,
                action=ACTION_PLUS_X,
                reward=-0.5,
                cursor_before=(0.1, 0.1, 0.1),
                cursor_after=(0.1, 0.1, 0.1),
                done=False,
                collision=True,
            ),
            ObservationTraceStep(
                step=1,
                observation=background,
                action=ACTION_PLUS_Z,
                reward=-0.05,
                cursor_before=(0.1, 0.1, 0.1),
                cursor_after=(0.1, 0.1, 0.13),
                done=False,
                collision=False,
            ),
        ]
    )


def test_backend_explains_visible_collision_without_cli_or_checkpoint(tmp_path: Path):
    observation = collision_observation()
    request = ExplanationRequest(
        run_ref="mock:smoke",
        checkpoint="mock",
        trajectory="in-memory",
        method="occlusion",
        focus="collisions",
    )

    report = explain_policy(
        request,
        schema=FeatureSchema.from_observation(observation),
        trace=trace_with_visible_collision(observation),
        scorer=LinearMockPolicyScorer(),
        output_dir=tmp_path,
    )

    assert tmp_path.joinpath("report.json").exists()
    assert tmp_path.joinpath("collision_steps.csv").exists()
    assert len(report.steps) == 1
    assert report.steps[0].collision_visible is True
    assert report.steps[0].chosen_action == ACTION_PLUS_X
    assert report.steps[0].best_safe_action != ACTION_PLUS_X
    assert report.steps[0].score_margin < 0.0
    assert report.steps[0].group_attributions["chosen_ray_hit"] < 0.0
    assert report.steps[0].group_attributions["chosen_ray_type"] < 0.0
