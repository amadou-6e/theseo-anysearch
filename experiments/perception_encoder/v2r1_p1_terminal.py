"""Record the terminal v2r1 P1 state when P0D cannot proceed."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def blocked_p1_report(config: dict[str, Any], p0d: dict[str, Any]) -> dict[str, Any]:
    """Validate E2 and record that replacement P1 was not executed."""

    if config["run_id"] != "voxel-encoder-pilot-v2r1-p1-1":
        raise ValueError("unexpected replacement P1 run identity")
    if p0d["run_id"] != "voxel-encoder-pilot-v2r1-p0d-1":
        raise ValueError("unexpected prerequisite P0D run identity")
    if p0d["report_payload_sha256"] != config["p0d_report_payload_sha256"]:
        raise ValueError("P0D payload differs from the frozen E3 prerequisite")
    expected_decision = {
        "decision": "blocked",
        "reason": "no_retained_bundle",
        "detail": "E1 did not freeze a valid active denominator set",
        "next": None,
    }
    if p0d["status"] != "blocked" or p0d["decision"] != expected_decision:
        raise ValueError("replacement P1 requires the recorded no-retained-bundle block")
    report: dict[str, Any] = {
        "issue": 337,
        "run_id": config["run_id"],
        "status": "not_started",
        "disposition": "reject",
        "integration_base_sha": config["integration_base_sha"],
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": config["spec_commit"],
        "dataset_id": "voxel-encoder-pilot-v2r1-dataset-1",
        "prerequisite": {
            "run_id": p0d["run_id"],
            "status": p0d["status"],
            "decision_reason": p0d["decision"]["reason"],
            "report_payload_sha256": p0d["report_payload_sha256"],
        },
        "execution": {
            "trials_started": 0,
            "optimizer_updates": 0,
            "accelerator_hours": 0.0,
            "new_training_results": False,
        },
        "decision": {
            "decision": "blocked",
            "reason": "p0d_no_retained_bundle",
            "detail": "No calibrated objective bundle was eligible for a replacement P1 screen",
            "next": None,
        },
        "v1_p1_supersede": {
            "program": "voxel-encoder-pilot-v1",
            "run_id": "voxel-encoder-pilot-v1-p1-1",
            "prior_decision": "no_viable_direction",
            "status": "superseded",
            "reason": "calibration_contract_invalidated",
            "fired_veto": "false_open_rate_max",
            "evidence_report_sha256": config["v1_p0c_report_payload_sha256"],
            "replacement_run_executed": False,
        },
    }
    report["report_payload_sha256"] = _canonical_sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    p0d = json.loads(Path(config["p0d_report_path"]).read_text(encoding="utf-8"))
    report = blocked_p1_report(config, p0d)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
