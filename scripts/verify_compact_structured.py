"""Replay the structured-code search on its original CUDA inference path."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.verify_compact_search import require
from theseo_anysearch.garden.pilots import compact_structured as study

prior = study.prior


def checked_report(directory):
    report = json.loads((directory / "report.json").read_text()); digest = report.pop("report_payload_sha256")
    require(study.d.base.payload_sha256(report) == digest, "report hash mismatch")
    study.check_registration(report["registration"])
    require([r["config"] for r in report["records"]] == study.configs(), "candidate set mismatch")
    for name, expected in report["artifacts"].items():
        require(Path(name).name == name, "unsafe artifact basename")
        require(prior.previous.file_hash(directory / name) == expected, f"artifact mismatch: {name}")
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
        require(record["shape"] == list(prior.cache_shape(row)), "cache shape mismatch")
        caches[split] = prior.open_cache(directory / name, record["shape"], record["sha256"])
        hidden = row["hidden"].reshape(-1, 33, 33, 33); bank = len(hidden) // len(row["targets"])
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
        require(saved["state"] == state and state["steps"] == study.PLAN["steps"], "stage state mismatch")
        require(state["sampled_queries"] == study.PLAN["steps"] * study.PLAN["batch"] * study.PLAN["queries"], "query exposure mismatch")
        model.load_state_dict(saved["model"])
        require(study.d.base.encoder_state_sha256(model) == state["model_state_sha256"], "model state mismatch")
        return model.eval().requires_grad_(False)

    def close(a, b, label):
        require(torch.allclose(a, b, rtol=1e-5, atol=1e-6), f"{label} replay mismatch")

    for record in report["records"]:
        config = record["config"]; number = config["id"]
        model = restore(f"train-{number}", study.models.make_model(config).cuda())
        require(study.d.base.encoder_state_sha256(model.aggregation) == record["aggregation_state_sha256"], "aggregation hash mismatch")
        stored = load(f"vectors-{number}.pt")
        require(stored["aggregation_hash"] == record["aggregation_state_sha256"], "vector key mismatch")
        for split in ("probe", "selection"):
            close(prior.vectors(model.aggregation, caches[split], float("inf")), stored[split], f"{split} vectors")
        artifact = load(f"assessment-{number}.pt")
        require(artifact["record"] == record, "assessment record mismatch")
        _, stats = study.models.normalized_vectors(stored["probe"], config)
        require(all(torch.equal(stats[k], artifact["statistics"][k]) for k in stats), "normalization mismatch")
        head = restore(f"probe-{number}", study.models.make_head(config).cuda())
        selected = study.heads.normalize(stored["selection"], stats)
        close(prior.predict(head, "vector", selected, float("inf")), artifact["prediction"], "independent head")
        require(prior.metrics(artifact["prediction"], rows["selection"]) == {k: record[k] for k in ("selection", "thresholds")}, "independent metrics mismatch")
        permutation = torch.randperm(len(selected), generator=torch.Generator().manual_seed(399))
        for name, values in (("shuffled", selected[permutation]), ("zeroed", torch.zeros_like(selected))):
            close(prior.predict(head, "vector", values, float("inf")), artifact[name], name)
            require(prior.diagnosis.compact_metrics(artifact[name], rows["selection"], record["thresholds"]) == record[name], f"{name} metrics mismatch")
        close(prior.predict(model, "volume", caches["selection"], float("inf")), artifact["trained_prediction"], "training decoder")
        require(prior.metrics(artifact["trained_prediction"], rows["selection"]) == record["trained_decoder"], "training decoder metrics mismatch")
        print(json.dumps({"verified_candidate": number}), flush=True)
        del model, head, stored
    spatial = restore("spatial", prior.models.SpatialReference().cuda())
    controls = [("spatial", spatial, "spatial", caches["selection"])]
    for number in (0, 8, 10):
        config = study.configs()[number]; name = f"null-{number}"
        controls.append((name, restore(name, study.models.make_head(config).cuda()), "vector",
                         torch.zeros(len(rows["selection"]["targets"]), config["dimension"])))
    for name, model, kind, inputs in controls:
        saved = load(f"{name}-selection.pt")
        close(prior.predict(model, kind, inputs, float("inf")), saved, name)
        require(prior.metrics(saved, rows["selection"]) == report["controls"][name], "control metrics mismatch")
    finalists = sorted(report["records"], key=study.corpus.rank, reverse=True)[:2]
    lock = {"identity_sha256": report["registration"]["identity_sha256"], "ids": [r["config"]["id"] for r in finalists]}
    require(json.loads((directory / "selection-lock.json").read_text()) == lock, "selection lock mismatch")
    require(report["selected_id"] == lock["ids"][0], "selected ID mismatch")
    require([r["id"] for r in report["common"]] == lock["ids"], "common-head identity mismatch")
    require([(r["id"], r["head"]) for r in report["development"]] == [(i, h) for i in lock["ids"] for h in ("native", "common")], "development identity mismatch")
    for record, common in zip(finalists, report["common"]):
        config = record["config"]; number = config["id"]
        model = restore(f"train-{number}", study.models.make_model(config).cuda())
        stats = load(f"assessment-{number}.pt")["statistics"]
        z = study.heads.normalize(prior.vectors(model.aggregation, caches["development"], float("inf")), stats)
        for family in ("native", "common"):
            stage = f"probe-{number}" if family == "native" else f"common-{number}"
            factory = study.models.make_head(config) if family == "native" else study.heads.ConvolutionalReadout(config["dimension"])
            head = restore(stage, factory.cuda())
            if family == "common":
                selected = study.heads.normalize(load(f"vectors-{number}.pt")["selection"], stats)
                saved = load(f"common-{number}-selection.pt")
                close(prior.predict(head, "vector", selected, float("inf")), saved, "common selection")
                require(prior.metrics(saved, rows["selection"]) == {k: common[k] for k in ("selection", "thresholds")}, "common metrics mismatch")
            saved = load(f"development-{number}-{family}.pt")
            close(prior.predict(head, "vector", z, float("inf")), saved, "development")
            assessment = next(r for r in report["development"] if r["id"] == number and r["head"] == family)
            thresholds = record["thresholds"] if family == "native" else common["thresholds"]
            require(prior.diagnosis.compact_metrics(saved, rows["development"], thresholds) == assessment["metrics"], "development metrics mismatch")
    saved = load("spatial-development.pt")
    close(prior.predict(spatial, "spatial", caches["development"], float("inf")), saved, "spatial development")
    require(prior.diagnosis.compact_metrics(saved, rows["development"], report["controls"]["spatial"]["thresholds"]) == report["spatial_development"], "spatial development metrics mismatch")
    for cache in caches.values():
        cache._mmap.close()
    return {"report_payload_sha256": digest, "artifacts_verified": len(report["artifacts"]),
            "backbone_batch_replays": 8, "candidate_replays": 12, "control_replays": 4,
            "common_head_replays": 2, "development_replays": 5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path); parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.source), indent=2))
