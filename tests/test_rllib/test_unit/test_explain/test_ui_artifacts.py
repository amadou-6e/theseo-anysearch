"""Tests for interactive explanation exports."""

import io
import zipfile

import numpy as np

from theseo_anysearch.rllib.explain.models import ExplainedStep, ExplanationReport
from theseo_anysearch.rllib.explain.ui.artifacts import build_artifact_bundle


def test_bundle_contains_reproducibility_artifacts() -> None:
    """The UI exports the same core artifacts as the command workflow."""

    report = ExplanationReport(
        run_ref="mock:run",
        checkpoint="latest",
        trajectory="interactive-observation",
        algorithm="dqn",
        score_type="q_value",
        method="occlusion",
        feature_schema_version=1,
        scenario_validity="not_environment_validated",
        steps=[
            ExplainedStep(
                step=0,
                chosen_action=0,
                chosen_direction=(-1, -1, -1),
                collision_visible=False,
                chosen_score=1.0,
                best_safe_action=1,
                best_safe_score=0.5,
                score_margin=0.5,
            )
        ],
    )
    payload = build_artifact_bundle(
        report,
        {"goal_distance": np.asarray([0.5], dtype=np.float32)},
        {
            "type": "observation",
            "chosen_action": "policy",
            "observation": {"goal_distance": [0.5]},
        },
    )

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "report.json", "summary.md", "steps.csv", "scenario.yaml",
            "request.yaml", "observations/step_000000.json",
        }
