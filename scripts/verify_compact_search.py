"""Replay completed compact-search evidence without fitting or selecting anew."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from theseo_anysearch.garden.pilots import compact_search as search
from theseo_anysearch.garden.pilots import compact_search_data as corpus


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    torch.set_num_threads(4)
    report = json.loads((directory / "report.json").read_text())
    digest = report.pop("report_payload_sha256")
    require(search.d.base.payload_sha256(report) == digest, "report hash mismatch")
    registration = report["registration"]
    require(search.d.base.payload_sha256(registration["payload"]) == registration["identity_sha256"],
            "registration hash mismatch")
    lock = json.loads((directory / "selection-lock.json").read_text())
    require(lock["registration_sha256"] == registration["identity_sha256"], "selection lock identity mismatch")
    finalists = sorted((r for r in report["records"] if r["steps"] == 8192), key=corpus.rank, reverse=True)
    require(len(finalists) == 2, "expected two final rungs")
    require(lock["finalists"] == [r["config"]["id"] for r in finalists], "finalist order mismatch")
    require(lock["winner"] == report["winner_selection_only"] == finalists[0]["config"]["id"],
            "selection winner mismatch")
    for name, expected in report["artifacts"].items():
        require(Path(name).name == name, "artifact path is not a basename")
        with (directory / name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(actual == expected, f"artifact hash mismatch: {name}")
    rows = corpus.data(["selection", "development"])
    require(corpus.identity(rows) == {s: registration["payload"]["data"][s] for s in rows}, "corpus identity mismatch")
    search.check_support(rows)
    for record in report["records"]:
        require(record["checkpoint"] in report["artifacts"], "unhashed checkpoint")
        saved = torch.load(directory / record["checkpoint"], map_location="cpu", weights_only=True)
        require(saved["record"] == record, "checkpoint record mismatch")
        prediction = saved["selection_predictions"]
        require(corpus.select_thresholds(prediction, rows["selection"]) == record["thresholds"], "threshold mismatch")
        require(corpus.evaluate(prediction, rows["selection"], record["thresholds"]) == record["selection"],
                "selection metric replay mismatch")
        del saved, prediction
    require([a["id"] for a in report["development"]] == lock["finalists"], "development finalist mismatch")
    for assessment, record in zip(report["development"], finalists):
        name = f"development-{assessment['id']}.pt"
        require(name in report["artifacts"], "unhashed development predictions")
        prediction = torch.load(directory / name, map_location="cpu", weights_only=True)
        require(corpus.evaluate(prediction, rows["development"], record["thresholds"]) == assessment["metrics"],
                "development metric replay mismatch")
    return {"verified_report_sha256": digest, "verified_artifacts": len(report["artifacts"]),
            "replayed_selection_stages": len(report["records"]), "replayed_development_candidates": len(finalists),
            "scope": "artifact identities and saved-prediction metric replay; no training reproduction"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
