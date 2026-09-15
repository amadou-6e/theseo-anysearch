"""Audit full-feature caches and replay the small-field reconstruction study."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.verify_compact_search import require
from theseo_anysearch.garden.pilots import compact_reconstruction as study


def checked_report(directory):
    report = json.loads((directory / "report.json").read_text()); digest = report.pop("report_payload_sha256")
    require(study.d.base.payload_sha256(report) == digest, "report hash mismatch")
    study.check_registration(report["registration"])
    require([r["config"] for r in report["records"]] == study.configs(), "candidate set mismatch")
    for name, expected in report["artifacts"].items():
        require(Path(name).name == name, "unsafe artifact basename")
        require(study.previous.file_hash(directory / name) == expected, f"artifact mismatch: {name}")
    return report, digest


def verify(directory, source):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    report, digest = checked_report(directory); payload = report["registration"]["payload"]
    rows = study.data(); study.corpus.support(rows)
    require(study.corpus.old.identity(rows) == payload["data"], "data mismatch")
    require(payload["source_encoders"] == study.corpus.old.helpers.prior.sources(source), "initializer mismatch")
    backbone = study.d.base.make_encoder(0, torch.device("cuda"))
    backbone.load_state_dict(torch.load(source / payload["source_encoders"][0]["path"], map_location="cuda", weights_only=True))
    backbone.eval().requires_grad_(False)
    backbone_hash = study.d.base.encoder_state_sha256(backbone)
    require(backbone_hash == payload["source_encoders"][0]["state_hash"], "backbone hash mismatch")
    caches = {}
    for split, row in rows.items():
        name = f"{split}-features.f32"; record = report["caches"][name]
        require(record["data"] == {split: payload["data"][split]}, "cache data key mismatch")
        require(record["backbone_hash"] == backbone_hash, "cache backbone key mismatch")
        require(record["sha256"] == report["artifacts"][name], "cache digest mismatch")
        require(record["shape"] == list(study.cache_shape(row)), "cache shape mismatch")
        caches[split] = study.open_cache(directory / name, record["shape"], record["sha256"])
        hidden = row["hidden"].reshape(-1, 33, 33, 33); bank = len(hidden) // len(row["targets"])
        # Re-extract complete original batches at both ends, not a different CUDA shape.
        for first in (0, len(hidden) - 16):
            ids = torch.arange(first, first + 16) // bank
            occ = row["occupancy"][ids].cuda().float(); mask = hidden[first:first + 16, None].cuda()
            with torch.no_grad():
                regenerated = backbone(study.d.base.VoxelLevel.from_occupancy(occ, unknown_mask=mask), mask).local_feature_volume.cpu()
            expected = torch.from_numpy(np.array(caches[split][first:first + 16], copy=True))
            require(torch.allclose(regenerated, expected, rtol=1e-5, atol=1e-6), "backbone cache replay mismatch")

    def load(name):
        require(name in report["artifacts"], "unhashed artifact")
        return torch.load(directory / name, map_location="cpu", weights_only=True)

    def restore(stage, model):
        state = report["stages"][stage]; saved = load(state["checkpoint"])
        require(saved["state"] == state and state["steps"] == 4096, "stage state mismatch")
        model.load_state_dict(saved["model"])
        require(study.d.base.encoder_state_sha256(model) == state["model_state_sha256"], "model state mismatch")
        return model.eval().requires_grad_(False)

    def close(a, b, label):
        require(torch.allclose(a, b, rtol=1e-5, atol=1e-6), f"{label} replay mismatch")

    for record in report["records"]:
        config = record["config"]; number = config["id"]
        model = restore(f"train-{number}", study.make_model(config))
        require(study.d.base.encoder_state_sha256(model.aggregation) == record["aggregation_state_sha256"], "aggregation hash mismatch")
        stored = load(f"vectors-{number}.pt")
        require(stored["aggregation_hash"] == record["aggregation_state_sha256"], "vector key mismatch")
        for split in ("probe", "selection"):
            close(study.vectors(model.aggregation, caches[split], float("inf")), stored[split], f"{split} vectors")
        artifact = load(f"assessment-{number}.pt")
        require(artifact["record"] == record, "assessment record mismatch")
        _, stats = study.heads.normalized_vectors(stored["probe"])
        require(all(torch.equal(stats[k], artifact["statistics"][k]) for k in stats), "normalization mismatch")
        head = restore(f"probe-{number}", study.heads.ConvolutionalReadout(config["dimension"]).cuda())
        selected = study.heads.normalize(stored["selection"], stats)
        prediction = study.predict(head, "vector", selected, float("inf"))
        close(prediction, artifact["prediction"], "independent head")
        require(study.metrics(artifact["prediction"], rows["selection"]) == {k: record[k] for k in ("selection", "thresholds")}, "independent metrics mismatch")
        permutation = torch.randperm(len(selected), generator=torch.Generator().manual_seed(397))
        shuffled = study.predict(head, "vector", selected[permutation], float("inf"))
        close(shuffled, artifact["shuffled"], "shuffle")
        require(study.diagnosis.compact_metrics(artifact["shuffled"], rows["selection"], record["thresholds"]) == record["shuffled"], "shuffle metrics mismatch")
        trained = study.predict(model, "volume", caches["selection"], float("inf"))
        close(trained, artifact["trained_prediction"], "training decoder")
        require(study.metrics(artifact["trained_prediction"], rows["selection"]) == record["trained_decoder"], "training decoder metrics mismatch")
        print(json.dumps({"verified_candidate": number}), flush=True)
        del model, head, stored
    spatial = restore("spatial", study.models.SpatialReference().cuda())
    null = restore("null", study.heads.ConvolutionalReadout().cuda())
    for name, model, kind, inputs in (("spatial", spatial, "spatial", caches["selection"]),
                                      ("null", null, "vector", torch.zeros(len(rows["selection"]["targets"]), 64))):
        saved = load(f"{name}-selection.pt")
        close(study.predict(model, kind, inputs, float("inf")), saved, name)
        require(study.metrics(saved, rows["selection"]) == report["controls"][name], "control metrics mismatch")
    finalists = sorted(report["records"], key=study.corpus.rank, reverse=True)[:2]
    lock = {"identity_sha256": report["registration"]["identity_sha256"], "ids": [r["config"]["id"] for r in finalists]}
    require(json.loads((directory / "selection-lock.json").read_text()) == lock, "selection lock mismatch")
    require(report["selected_id"] == lock["ids"][0], "selected ID mismatch")
    require([r["id"] for r in report["development"]] == lock["ids"], "development identity mismatch")
    for record, assessment in zip(finalists, report["development"]):
        config = record["config"]; number = config["id"]
        model = restore(f"train-{number}", study.make_model(config))
        head = restore(f"probe-{number}", study.heads.ConvolutionalReadout(config["dimension"]).cuda())
        stats = load(f"assessment-{number}.pt")["statistics"]
        z = study.heads.normalize(study.vectors(model.aggregation, caches["development"], float("inf")), stats)
        saved = load(f"development-{number}.pt")
        close(study.predict(head, "vector", z, float("inf")), saved, "development")
        require(study.diagnosis.compact_metrics(saved, rows["development"], record["thresholds"]) == assessment["metrics"], "development metrics mismatch")
    saved = load("spatial-development.pt")
    close(study.predict(spatial, "spatial", caches["development"], float("inf")), saved, "spatial development")
    require(study.diagnosis.compact_metrics(saved, rows["development"], report["controls"]["spatial"]["thresholds"]) == report["spatial_development"], "spatial development metrics mismatch")
    for cache in caches.values():
        cache._mmap.close()
    return {"report_payload_sha256": digest, "artifacts_verified": len(report["artifacts"]),
            "backbone_batch_replays": 8, "candidate_replays": 8, "control_replays": 2, "development_replays": 3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path); parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.source), indent=2))
