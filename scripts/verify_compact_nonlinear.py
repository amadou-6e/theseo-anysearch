"""Replay completed compact nonlinear heads on the original CUDA inference path."""
import argparse
import json
import os
from pathlib import Path

import torch

from scripts.verify_compact_search import require
from theseo_anysearch.garden.pilots import compact_nonlinear as experiment


def verify(directory):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    report = json.loads((directory / "report.json").read_text()); digest = report.pop("report_payload_sha256")
    require(experiment.d.base.payload_sha256(report) == digest, "report hash mismatch")
    registration = report["registration"]
    require(experiment.d.base.payload_sha256(registration["payload"]) == registration["identity_sha256"], "registration mismatch")
    for name, expected in report["artifacts"].items():
        require(Path(name).name == name, "unsafe artifact basename")
        require(experiment.previous.file_hash(directory / name) == expected, f"artifact mismatch: {name}")
    require(len(report["records"]) == 34, "expected thirty-four completed evaluations")
    configs = [c for c in experiment.configs() if c["mode"] == "candidate"]
    best = [max((r for r in report["records"] if r["config"] == c), key=experiment.rank) for c in configs]
    finalists = sorted(best, key=experiment.rank, reverse=True)[:2]
    lock = json.loads((directory / "selection-lock.json").read_text())
    require(lock == {"identity_sha256": registration["identity_sha256"], "checkpoints": [r["checkpoint"] for r in finalists]}, "selection lock mismatch")
    require(report["selected_checkpoint"] == finalists[0]["checkpoint"], "winner mismatch")
    rows = experiment.data(["selection", "calibration", "development"])
    require(experiment.corpus.old.identity(rows) == {s: registration["payload"]["data"][s] for s in rows}, "data mismatch")
    cache = torch.load(directory / "vectors.pt", map_location="cpu", weights_only=True)
    require(cache["identity"] == {s: registration["payload"]["data"][s] for s in ("probe", "selection", "calibration")}, "cache identity mismatch")
    require(cache["sources"] == registration["payload"]["sources"], "cached encoder mismatch")
    require(cache["backbone_hash"] == registration["payload"]["source_encoders"][0]["state_hash"], "cached backbone mismatch")
    statistics = [experiment.heads.normalized_vectors(z)[1] for z in cache["vectors"]["probe"]]
    for record in report["records"]:
        require(record["checkpoint"] in report["artifacts"], "unhashed checkpoint")
        saved = torch.load(directory / record["checkpoint"], map_location="cpu", weights_only=True)
        require(saved["record"] == record, "checkpoint record mismatch")
        head = experiment.heads.make_readout(record["config"]["kind"]).cuda().requires_grad_(False)
        head.load_state_dict(saved["head"])
        require(experiment.d.base.encoder_state_sha256(head) == record["head_state_sha256"], "head state mismatch")
        oracle = record["config"]["mode"] == "oracle"
        if oracle:
            vectors = torch.zeros(4, 64); vectors[:, :4] = torch.eye(4)
            targets = rows["calibration"]["targets"][[1, 4, 7, 10]]
            require(experiment.oracle_scores(saved["prediction"], targets) == record["oracle"], "oracle metric mismatch")
        else:
            source = record["config"]["source"]
            require(all(torch.equal(saved["normalization"][k], statistics[source][k]) for k in ("mean", "scale")), "normalization mismatch")
            vectors = experiment.heads.normalize(cache["vectors"]["selection"][source], saved["normalization"])
            if record["config"]["mode"] == "null":
                vectors = torch.zeros_like(vectors)
            require(experiment.corpus.old.select_thresholds(saved["prediction"], rows["selection"]) == record["thresholds"], "threshold mismatch")
            require(experiment.diagnosis.compact_metrics(saved["prediction"], rows["selection"], record["thresholds"]) == record["selection"], "selection metric mismatch")
        regenerated = experiment.predict(head, vectors, float("inf"))
        require(torch.allclose(regenerated, saved["prediction"], rtol=1e-5, atol=1e-6), "original-device head prediction mismatch")
        if record["config"]["mode"] == "candidate":
            permutation = torch.randperm(len(vectors), generator=torch.Generator().manual_seed(395))
            shuffled = experiment.predict(head, vectors[permutation], float("inf"))
            require(experiment.diagnosis.compact_metrics(shuffled, rows["selection"], record["thresholds"]) == record["shuffled"], "shuffled metric mismatch")
    for control in report["ridge_controls"]:
        source = control["source"]
        saved = torch.load(directory / f"ridge-{source}.pt", map_location="cuda", weights_only=True)
        regenerated = experiment.corpus.old.ridge_predict(saved["readout"], cache["vectors"]["selection"][source].cuda()).cpu()
        require(torch.allclose(regenerated, saved["prediction"].cpu(), rtol=1e-5, atol=1e-6), "ridge prediction mismatch")
        require(experiment.diagnosis.compact_metrics(saved["prediction"].cpu(), rows["selection"], control["thresholds"]) == control["selection"], "ridge metric mismatch")
    require([r["checkpoint"] for r in report["development"]] == lock["checkpoints"], "development identity mismatch")
    for assessment, record in zip(report["development"], finalists):
        name = f"development-{record['config']['id']}.pt"
        require(name in report["artifacts"], "unhashed development predictions")
        prediction = torch.load(directory / name, map_location="cpu", weights_only=True)
        require(experiment.diagnosis.compact_metrics(prediction, rows["development"], record["thresholds"]) == assessment["metrics"], "development metric mismatch")
    return {"report_payload_sha256": digest, "artifacts_verified": len(report["artifacts"]),
            "original_device_head_replays": 34, "shuffle_replays": 24, "ridge_replays": 2, "development_metric_replays": 2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
