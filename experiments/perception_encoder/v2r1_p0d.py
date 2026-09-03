"""Execute or block the v2r1 four-times-density P0D transition."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def blocked_p0d_report(config: dict[str, Any], p0c: dict[str, Any]) -> dict[str, Any]:
    """Derive the required blocked decision without opening P0D observations."""

    if config["run_id"] != "voxel-encoder-pilot-v2r1-p0d-1":
        raise ValueError("unexpected P0D run identity")
    if p0c["run_id"] != "voxel-encoder-pilot-v2r1-p0c-1":
        raise ValueError("unexpected prerequisite P0C run identity")
    if p0c["report_payload_sha256"] != config["p0c_report_payload_sha256"]:
        raise ValueError("P0C payload differs from the frozen E2 prerequisite")
    if p0c["status"] != "blocked":
        raise ValueError("this transition only records a P0C-blocked P0D outcome")
    report: dict[str, Any] = {
        "issue": 335,
        "run_id": config["run_id"],
        "status": "blocked",
        "disposition": "reject",
        "integration_base_sha": config["integration_base_sha"],
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": config["spec_commit"],
        "dataset_id": "voxel-encoder-pilot-v2r1-dataset-1",
        "prerequisite": {
            "run_id": p0c["run_id"],
            "status": p0c["status"],
            "report_payload_sha256": p0c["report_payload_sha256"],
            "denominator_failures": sorted(p0c["denominator_failures"]),
        },
        "observation_density_multiplier": 4,
        "observations_opened": False,
        "trials_started": 0,
        "accelerator_hours": 0.0,
        "decision": {
            "decision": "blocked",
            "reason": "no_retained_bundle",
            "detail": "E1 did not freeze a valid active denominator set",
            "next": None,
        },
        "replacement_p1": {
            "run_id": "voxel-encoder-pilot-v2r1-p1-1",
            "status": "not_started",
            "blocked_by": config["run_id"],
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
    p0c = json.loads(Path(config["p0c_report_path"]).read_text(encoding="utf-8"))
    report = blocked_p0d_report(config, p0c)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
