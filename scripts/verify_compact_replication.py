"""Replay the complete fine729 replication on the original CUDA path."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.verify_compact_search import require
from theseo_anysearch.garden.pilots import compact_replication as study


def verify(directory, source):
    study.configure_cuda()
    report = json.loads((directory / "report.json").read_text()); digest = report.pop("report_payload_sha256")
    require(study.d.base.payload_sha256(report) == digest, "report hash mismatch")
    payload = study.check_registration(report["registration"])
    require([r["seed"] for r in report["records"]] == list(study.SEEDS), "seed records mismatch")
    for name, expected in report["artifacts"].items():
        require(Path(name).name == name, "unsafe artifact name")
        require(study.prior.previous.file_hash(directory / name) == expected, "artifact hash mismatch")
    require(report["assessment"] == study.assess(report["development"]), "assessment mismatch")
    require(json.loads((directory / "selection-lock.json").read_text()) == study.lock_thresholds(report["registration"], report["records"]), "threshold lock mismatch")
    rows = study.data(); study.corpus.support(rows)
    require(study.corpus.old.identity(rows) == payload["data"], "data mismatch")
    require(payload["source_encoders"] == study.corpus.old.helpers.prior.sources(source), "initializer mismatch")
    backbone = study.d.base.make_encoder(0, torch.device("cuda"))
    backbone.load_state_dict(torch.load(source / payload["source_encoders"][0]["path"], map_location="cuda", weights_only=True))
    backbone.eval().requires_grad_(False)
    h = study.d.base.encoder_state_sha256(backbone)
    require(h == payload["source_encoders"][0]["state_hash"], "backbone mismatch")

    def close(a, b):
        require(torch.allclose(a, b, rtol=1e-5, atol=1e-6), "prediction replay mismatch")

    caches = {}
    for split, row in rows.items():
        name = f"{split}-features.f32"; record = report["caches"][name]
        require(record["data"] == {split: payload["data"][split]} and record["backbone_hash"] == h, "cache key mismatch")
        require(record["sha256"] == report["artifacts"][name] and record["shape"] == list(study.prior.cache_shape(row)), "cache metadata mismatch")
        caches[split] = study.prior.open_cache(directory / name, record["shape"], record["sha256"])
        hidden = row["hidden"].reshape(-1, 33, 33, 33); bank = len(hidden) // len(row["targets"])
        for first in (0, len(hidden) - 16):
            ids = torch.arange(first, first + 16) // bank
            occ = row["occupancy"][ids].cuda().float(); mask = hidden[first:first + 16, None].cuda()
            with torch.no_grad():
                features = backbone(study.d.base.VoxelLevel.from_occupancy(occ, unknown_mask=mask), mask).local_feature_volume.cpu()
            close(features, torch.from_numpy(np.array(caches[split][first:first + 16], copy=True)))

    def load(name):
        require(name in report["artifacts"], "unhashed artifact")
        return torch.load(directory / name, map_location="cpu", weights_only=True)

    def restore(stage, model):
        state = report["stages"][stage]; saved = load(state["checkpoint"])
        require(saved["state"] == state and state["steps"] == study.PLAN["steps"], "stage mismatch")
        require(state["sampled_queries"] == study.PLAN["steps"] * study.PLAN["batch"] * study.PLAN["queries"], "query exposure mismatch")
        model.load_state_dict(saved["model"])
        require(study.d.base.encoder_state_sha256(model) == state["model_state_sha256"], "model state mismatch")
        return model.eval().requires_grad_(False)

    for record in report["records"]:
        seed = record["seed"]
        model = restore(f"train-{seed}", study.make_model(seed).cuda())
        require(study.d.base.encoder_state_sha256(model.aggregation) == record["aggregation_state_sha256"], "aggregation hash mismatch")
        stored = load(f"vectors-{seed}.pt")
        require(stored["aggregation_hash"] == record["aggregation_state_sha256"], "vector key mismatch")
        for split in ("probe", "selection"):
            close(study.prior.vectors(model.aggregation, caches[split], float("inf")), stored[split])
        saved = load(f"assessment-{seed}.pt")
        require(saved["record"] == record, "selection record mismatch")
        _, stats = study.parent.models.normalized_vectors(stored["probe"], study.CONFIG)
        require(all(torch.equal(stats[k], saved["statistics"][k]) for k in stats), "statistics mismatch")
        native = restore(f"probe-{seed}", study.make_head(seed).cuda())
        spatial = restore(f"spatial-{seed}", study.make_spatial(seed).cuda())
        null = restore(f"null-{seed}", study.make_head(seed).cuda())
        for split in ("selection", "development"):
            z = study.parent.heads.normalize(stored[split] if split == "selection" else study.prior.vectors(model.aggregation, caches[split], float("inf")), stats)
            permutation = torch.randperm(len(z), generator=torch.Generator().manual_seed(seed + 30000))
            artifact = saved if split == "selection" else load(f"development-{seed}.pt")
            for key, head, kind, values in (("native", native, "vector", z),
                                            ("shuffled", native, "vector", z[permutation]),
                                            ("zeroed", native, "vector", torch.zeros_like(z)),
                                            ("null", null, "vector", torch.zeros_like(z)),
                                            ("spatial", spatial, "spatial", caches[split])):
                prediction = artifact["predictions"][key]
                close(study.prior.predict(head, kind, values, float("inf")), prediction)
                if split == "selection" and key in ("native", "spatial", "null"):
                    require(study.prior.metrics(prediction, rows[split]) == record[key], "calibration mismatch")
                else:
                    metrics = study.prior.diagnosis.compact_metrics(prediction, rows[split], record[key if key in ("spatial", "null") else "native"]["thresholds"])
                    require(metrics == artifact["record"][key], "metrics mismatch")
            if split == "development":
                require(artifact["record"] == next(r for r in report["development"] if r["seed"] == seed), "development record mismatch")
        print(json.dumps({"verified_seed": seed}), flush=True)
    for cache in caches.values():
        cache._mmap.close()
    return {"report_payload_sha256": digest, "artifacts_verified": len(report["artifacts"]),
            "seed_replays": 3, "prediction_replays": 30, "backbone_batch_replays": 8}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path); parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.source), indent=2))
