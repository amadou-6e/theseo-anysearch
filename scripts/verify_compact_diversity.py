"""Verify completed diversity-study hashes, ranking and saved prediction metrics."""
import argparse
import json
import os
from pathlib import Path

import torch

from scripts.verify_compact_search import require
from theseo_anysearch.garden.pilots import compact_diversity as experiment
from theseo_anysearch.garden.pilots import compact_diversity_data as corpus


def verify(directory, device="cuda"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    report = json.loads((directory / "report.json").read_text())
    digest = report.pop("report_payload_sha256")
    require(experiment.d.base.payload_sha256(report) == digest, "report hash mismatch")
    registration = report["registration"]
    require(experiment.d.base.payload_sha256(registration["payload"]) == registration["identity_sha256"], "registration mismatch")
    for name, expected in report["artifacts"].items():
        require(Path(name).name == name, "unsafe artifact basename")
        require(experiment.file_hash(directory / name) == expected, f"artifact mismatch: {name}")
    require(len(report["records"]) == 18, "expected eighteen completed stages")
    best = [max((r for r in report["records"] if r["config"] == c), key=corpus.rank) for c in experiment.configs()]
    finalists = sorted(best, key=corpus.rank, reverse=True)[:2]
    lock = json.loads((directory / "selection-lock.json").read_text())
    require(lock == {"identity_sha256": registration["identity_sha256"], "checkpoints": [r["checkpoint"] for r in finalists]}, "selection lock mismatch")
    require(report["selected_checkpoint"] == finalists[0]["checkpoint"], "winner mismatch")
    rows = corpus.data(["selection", "development"])
    require(corpus.old.identity(rows) == {s: registration["payload"]["data"][s] for s in rows}, "data identity mismatch")
    cache = torch.load(directory / "features.pt", map_location="cpu", weights_only=True)
    require(cache["backbone_hash"] == registration["payload"]["source_encoders"][0]["state_hash"], "cache backbone mismatch")
    require(cache["identity"] == {s: registration["payload"]["data"][s] for s in ("train", "probe", "selection")}, "cache identity mismatch")
    for record in report["records"]:
        require(record["checkpoint"] in report["artifacts"], "unhashed checkpoint")
        saved = torch.load(directory / record["checkpoint"], map_location=device, weights_only=True)
        require(saved["record"] == record, "checkpoint record mismatch")
        prediction = saved["selection_predictions"].cpu()
        require(corpus.old.select_thresholds(prediction, rows["selection"]) == record["thresholds"], "threshold replay mismatch")
        require(corpus.old.evaluate(prediction, rows["selection"], record["thresholds"]) == record["selection"], "selection replay mismatch")
        aggregation = corpus.aggregation(record["config"]["side"]).to(device).eval().requires_grad_(False)
        aggregation.load_state_dict(saved["aggregation"])
        require(experiment.d.base.encoder_state_sha256(aggregation) == record["aggregation_state_sha256"], "aggregation state mismatch")
        with torch.no_grad():
            z = aggregation(cache["features"]["selection"][record["config"]["side"]].to(device))
            regenerated = corpus.old.ridge_predict(saved["readout"], z).cpu()
        require(torch.allclose(regenerated, prediction, rtol=1e-5, atol=1e-6), "cache-to-prediction replay mismatch")
    require([r["checkpoint"] for r in report["development"]] == lock["checkpoints"], "development identities mismatch")
    for assessment, record in zip(report["development"], finalists):
        name = f"development-{record['config']['id']}.pt"
        require(name in report["artifacts"], "unhashed development predictions")
        prediction = torch.load(directory / name, map_location="cpu", weights_only=True)
        require(corpus.old.evaluate(prediction, rows["development"], record["thresholds"]) == assessment["metrics"], "development replay mismatch")
    return {"report_payload_sha256": digest, "artifacts_verified": len(report["artifacts"]),
            "selection_stages_replayed": 18, "cache_to_prediction_replays": 18, "development_replays": 2,
            "inference_device": device}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.device), indent=2))
