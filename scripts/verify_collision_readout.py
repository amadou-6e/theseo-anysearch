"""Full original-device evidence replay for the frozen collision-readout study."""
import argparse
import json
from pathlib import Path
import time

import torch

from scripts import verify_compact_validation as prior
from theseo_anysearch.garden.pilots import compact_collision_readout as study


@torch.no_grad()
def native_inputs(report, rows, output, package_root, seed, memo, deadline):
    encoder, decoder, _ = study.common.package.load_compact_package(package_root, seed=seed, device="cuda")
    del encoder
    decoder.eval().requires_grad_(False)
    before = study.common.package.encoder_state_sha256(decoder)
    expected = {"before": before, "after": before, "frozen_parameters": sum(p.numel() for p in decoder.parameters())}
    if report["native_states"][str(seed)] != expected: raise ValueError("native decoder state mismatch")
    names = {f"inputs-{split}-native-{seed}.pt": memo[f"inputs-{split}-compact-{seed}.pt"].flatten(1) for split in rows}
    names[f"native-zero-{seed}.pt"] = torch.zeros(1, 729)
    for name, code in names.items():
        actual = study.common.prior.predict(decoder, "vector", code, deadline).reshape(-1, 3, 17, 17, 17)
        saved = prior.load(output, report, name)
        prior.same_prediction(actual, saved); memo[name] = saved
    if study.common.package.encoder_state_sha256(decoder) != before: raise ValueError("native replay mutated state")


def verify(env, rows, output, package_root):
    start = time.monotonic(); report = prior.checked_report(output, env)
    study.common.replication.configure_cuda(); deadline = time.monotonic() + 1800
    queries = prior.audit_queries(rows)
    lock = json.loads((output / "selection-lock.json").read_text())
    expected_lock = {"identity_sha256": env["identity_sha256"], "thresholds":
                     {f"{r['seed']}-{r['kind']}": r["thresholds"] for r in report["calibration"]}}
    if lock != expected_lock or len(lock["thresholds"]) != 21: raise ValueError("calibration lock mismatch")
    records = []; calibrations = []; predictions = {}; backbones = set(); batches = 0; replays = 0
    for seed in study.PLAN["seeds"]:
        memo, count, backbone = prior.verify_inputs(report, rows, output, package_root, seed, deadline)
        batches += count; backbones.add(backbone)
        if len(backbones) != 1: raise ValueError("shared spatial backbones differ")
        native_inputs(report, rows, output, package_root, seed, memo, deadline)
        for kind in study.PLAN["representations"]:
            stage = report["stages"][f"head-{seed}-{kind}"]
            saved = prior.load(output, report, stage["checkpoint"])
            if saved["state"] != stage or stage["steps"] != 4096 or stage["sampled_paths"] != 524288:
                raise ValueError("training exposure mismatch")
            model = study.make_head(seed, kind).cuda(); model.load_state_dict(saved["model"])
            if study.common.package.encoder_state_sha256(model) != stage["model_state_sha256"]: raise ValueError("head hash mismatch")
            calibrated = prior.load(output, report, f"calibration-{seed}-{kind}.pt")
            prediction = study.prior.predict(model, study.values(memo, "calibration", kind, seed), rows["calibration"], deadline)
            prior.same_prediction(prediction, calibrated["prediction"]); replays += 1
            cuts = study.thresholds(calibrated["prediction"], rows["calibration"])
            record = {"seed": seed, "kind": kind, "thresholds": cuts,
                      "metrics": study.metrics(calibrated["prediction"], rows["calibration"], cuts),
                      "parameters": sum(p.numel() for p in model.parameters())}
            if record != calibrated["record"] or cuts != lock["thresholds"][f"{seed}-{kind}"]:
                raise ValueError("calibration evidence mismatch")
            calibrations.append(record)
            for split in ("test", "ood"):
                for variant, grids in study.variants(memo, split, kind, seed).items():
                    saved = prior.load(output, report, f"test-{seed}-{variant}-{split}.pt")
                    prediction = study.prior.predict(model, grids, rows[split], deadline)
                    prior.same_prediction(prediction, saved["prediction"]); replays += 1
                    record = {"seed": seed, "kind": variant, "split": split, "thresholds": cuts,
                              "metrics": study.metrics(saved["prediction"], rows[split], cuts)}
                    if record != saved["record"]: raise ValueError("test metrics mismatch")
                    records.append(record); predictions[seed, variant, split] = saved["prediction"]
            del model
        del memo
        print(json.dumps({"readout_seed_verified": seed}), flush=True)
    contrasts = study.comparisons(predictions, rows)
    if records != report["records"] or calibrations != report["calibration"] or contrasts != report["comparisons"]:
        raise ValueError("aggregate evidence mismatch")
    if study.assessment(records, contrasts) != report["assessment"]: raise ValueError("assessment mismatch")
    result = {"verified": True, "report_payload_sha256": report["report_payload_sha256"],
              "artifacts_verified": len(report["artifacts"]), "heads_verified": 21,
              "prediction_groups_replayed": replays, "encoder_batches_replayed": batches,
              "queries_replayed": queries, "native_decoders_verified": 3, "elapsed_seconds": time.monotonic()-start}
    study.common.base.write_json(output / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "data-root", "package", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    env, rows = study.registration(args.registration, args.data_root)
    verify(env, rows, args.output, args.package)
