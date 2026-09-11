"""In-memory export bundle for interactive explanations."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Mapping

import numpy as np
import yaml

from theseo_anysearch.rllib.explain.models import ExplanationReport


def build_artifact_bundle(
    report: ExplanationReport,
    observation: Mapping[str, np.ndarray],
    scenario: dict,
) -> bytes:
    """Return a CLI-compatible explanation artifact ZIP archive."""

    step = report.steps[0]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "step", "chosen_action", "chosen_score", "best_safe_action",
            "best_safe_score", "score_margin", "collision_visible",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "step": step.step,
            "chosen_action": step.chosen_action,
            "chosen_score": step.chosen_score,
            "best_safe_action": step.best_safe_action,
            "best_safe_score": step.best_safe_score,
            "score_margin": step.score_margin,
            "collision_visible": step.collision_visible,
        }
    )
    strongest = max(
        step.group_attributions,
        key=lambda name: abs(step.group_attributions[name]),
        default="none",
    )
    summary = (
        "# Interactive policy explanation\n\n"
        f"The policy chose action `{step.chosen_action}` {step.chosen_direction}. "
        f"Its score margin over safe action `{step.best_safe_action}` was "
        f"`{step.score_margin:.6g}`. Strongest measured group: `{strongest}`.\n"
    )
    request = {
        "checkpoint": report.checkpoint,
        "source": {"scenario": "scenario.yaml"},
        "explanation": {
            "method": report.method,
            "focus": "all",
            "max_steps": 1,
            "background": "trace",
        },
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", json.dumps(report.to_json_dict(), indent=2))
        archive.writestr("summary.md", summary)
        archive.writestr("steps.csv", csv_buffer.getvalue())
        archive.writestr("scenario.yaml", yaml.safe_dump(scenario, sort_keys=False))
        archive.writestr("request.yaml", yaml.safe_dump(request, sort_keys=False))
        archive.writestr(
            "observations/step_000000.json",
            json.dumps(
                {name: np.asarray(value).tolist() for name, value in observation.items()},
                indent=2,
            ),
        )
    return buffer.getvalue()
