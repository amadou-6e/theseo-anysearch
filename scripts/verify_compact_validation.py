"""Replay compact stress and collision-transfer evidence on the original CUDA path."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from theseo_anysearch.garden.pilots import compact_validation as common
from theseo_anysearch.garden.pilots import compact_collision_transfer as transfer


def checked_report(output, env):
    report = json.loads((output / "report.json").read_text())
    digest = report.pop("report_payload_sha256")
    if common.base.payload_sha256(report) != digest or report["registration"] != env or report["status"] != "completed":
        raise ValueError("report identity mismatch")
    for name, sha in report["artifacts"].items():
        if Path(name).name != name or common.package.file_sha(output / name) != sha:
            raise ValueError("artifact integrity mismatch")
    report["report_payload_sha256"] = digest
    return report


def same_prediction(actual, saved):
    if actual.shape != saved.shape or not torch.isfinite(actual).all():
        raise ValueError("invalid replay prediction")
    torch.testing.assert_close(actual, saved, rtol=1e-5, atol=1e-6)


def audit_queries(rows):
    """Check saved query labels and visible-only starts independently of the head."""
    checked = 0
    parents = set()
    for row in rows.values():
        if parents.intersection(row["parents"]) or len(set(row["parents"])) != len(row["parents"]):
            raise ValueError("query corpus parent overlap")
        parents.update(row["parents"])
        occupancy = common.data.crop(row["occupancy"], 17)
        hidden = common.data.crop(row["hidden"], 17)
        for i, gid in enumerate(row["ids"]):
            paths = row["paths"][i]; valid = row["valid"][i]
            expected_paths, expected_valid = common.data.paths(gid, occupancy[i].numpy(), hidden[i].numpy())
            if not np.array_equal(expected_paths, paths.numpy()) or not np.array_equal(expected_valid, valid.numpy()):
                raise ValueError("query sampling replay mismatch")
            x, y, z = paths.unbind(-1)
            occupied = occupancy[i, x, y, z]; unknown = hidden[i, x, y, z]
            expected = {"labels": (occupied & valid).any(1),
                        "visible_hit": (occupied & ~unknown & valid).any(1),
                        "unknown_path": (unknown & valid).any(1)}
            if occupied[:, 0].any() or unknown[:, 0].any():
                raise ValueError("query start is not visibly free")
            if any(not torch.equal(value, row[name][i]) for name, value in expected.items()):
                raise ValueError("collision query labels mismatch")
            checked += len(paths)
    return checked


def load(output, report, name):
    if name not in report["artifacts"]:
        raise ValueError("untracked replay artifact")
    return torch.load(output / name, map_location="cpu", weights_only=True)


def verify_robustness(report, rows, output, package_root, deadline):
    records = []; replayed = 0
    for seed in common.PLANS["robustness"]["seeds"]:
        encoder, head, metadata = common.package.load_compact_package(package_root, seed=seed, device="cuda")
        before = common.package.encoder_state_sha256(encoder)
        head_before = common.package.encoder_state_sha256(head)
        expected = {"before": before, "after": before, "head": head_before}
        if report["encoder_states"][str(seed)] != expected:
            raise ValueError("frozen state mismatch")
        for index, (key, row) in enumerate(rows.items()):
            saved = load(output, report, f"robust-{seed}-{index}.pt")
            prediction = common.robust_prediction(encoder, head, row, deadline)
            same_prediction(prediction, saved["prediction"])
            record = {"seed": seed, "key": key,
                      "metrics": common.geometry_metrics(saved["prediction"], row, metadata["artifacts"][str(seed)]["thresholds"]),
                      "actual_density_mean": float(np.mean(row["actual_density"]))}
            if record != saved["record"]:
                raise ValueError("robustness metric mismatch")
            records.append(record); replayed += 1
        if common.package.encoder_state_sha256(encoder) != before or common.package.encoder_state_sha256(head) != head_before:
            raise ValueError("replay modified weights")
        print(json.dumps({"robustness_seed_verified": seed}), flush=True)
        del encoder, head
    if records != report["records"]:
        raise ValueError("robustness records mismatch")
    failed = []; unavailable = []; comparisons = []
    for record in records:
        row = rows[record["key"]]
        for task, target in common.data.prior.old.BARS.items():
            value = record["metrics"][task]["score"]
            if value is None:
                unavailable.append({"seed": record["seed"], "key": record["key"], "task": task})
            elif (value < target if task in ("occupied_iou", "boundary_f1") else value > target):
                failed.append({"seed": record["seed"], "key": record["key"], "task": task, "score": value, "target": target})
        if row["condition"] != "baseline":
            reference = next(r for r in records if r["seed"] == record["seed"] and r["key"] == f"baseline/{row['family']}")
            comparisons.append({"seed": record["seed"], "key": record["key"], "negative_distance_error_difference":
                common.bootstrap(-np.array(record["metrics"]["all_free_distance"]["per_geometry"]),
                                 -np.array(reference["metrics"]["all_free_distance"]["per_geometry"]), row["ids"],
                                 [row["family"]] * len(row["ids"]), [str(row["density"])] * len(row["ids"]))})
    if failed != report["target_failures"] or unavailable != report["unavailable"] or comparisons != report["comparisons"]:
        raise ValueError("robustness assessment mismatch")
    return {"prediction_groups_replayed": replayed}


@torch.no_grad()
def verify_inputs(report, rows, output, package_root, seed, deadline):
    encoder, head, _ = common.package.load_compact_package(package_root, seed=seed, device="cuda")
    del head
    before = common.package.encoder_state_sha256(encoder)
    backbone_sha = common.package.encoder_state_sha256(encoder.backbone)
    if report["encoder_states"][str(seed)] != {"before": before, "after": before}:
        raise ValueError("transfer frozen state mismatch")
    memo = {}; batches = 0
    for split, row in rows.items():
        names = {"compact": f"inputs-{split}-compact-{seed}.pt", "raw": f"inputs-{split}-raw.pt", "spatial": f"inputs-{split}-spatial.pt"}
        for name in names.values():
            memo[name] = load(output, report, name)
        for first in range(0, len(row["ids"]), 16):
            common.replication.d.check_deadline(deadline)
            occ = row["occupancy"][first:first+16].cuda(); mask = row["hidden"][first:first+16].cuda()[:, None]
            actual = encoder(occ, mask).reshape(-1, 1, 9, 9, 9).cpu()
            same_prediction(actual, memo[names["compact"]][first:first+16])
            # Shared raw/spatial caches have the same backbone for all packaged seeds.
            if seed == 401:
                level = common.base.VoxelLevel.from_occupancy(occ.float(), unknown_mask=mask)
                same_prediction(level.features[:, [0, 2]].cpu(), memo[names["raw"]][first:first+16])
                same_prediction(encoder.backbone(level, mask).local_feature_volume.cpu(), memo[names["spatial"]][first:first+16])
            batches += 1
    if common.package.encoder_state_sha256(encoder) != before:
        raise ValueError("input replay changed encoder")
    del encoder
    return memo, batches, backbone_sha


def verify_transfer(report, rows, output, package_root, deadline):
    queries_checked = audit_queries(rows)
    lock = json.loads((output / "selection-lock.json").read_text())
    expected_lock = {"identity_sha256": report["registration"]["identity_sha256"],
                     "thresholds": {f"{r['seed']}-{r['kind']}": r["threshold"] for r in report["calibration"]}}
    if lock != expected_lock or len(lock["thresholds"]) != 12:
        raise ValueError("calibration lock mismatch")
    results = []; calibrations = []; predictions = {}; batches = 0; replays = 0; backbone_hashes = set()
    for seed in transfer.PLAN["seeds"]:
        memo, count, backbone_sha = verify_inputs(report, rows, output, package_root, seed, deadline); batches += count
        backbone_hashes.add(backbone_sha)
        if len(backbone_hashes) != 1:
            raise ValueError("shared spatial cache requires identical packaged backbones")
        for kind in transfer.PLAN["representations"]:
            stage = report["stages"][f"head-{seed}-{kind}"]
            checkpoint = load(output, report, stage["checkpoint"])
            if checkpoint["state"] != stage or stage["steps"] != 4096 or stage["sampled_paths"] != 4096 * 128:
                raise ValueError("training exposure mismatch")
            model = transfer.make_head(seed, kind).cuda(); model.load_state_dict(checkpoint["model"])
            if common.package.encoder_state_sha256(model) != stage["model_state_sha256"]:
                raise ValueError("head checkpoint mismatch")
            cal = load(output, report, f"calibration-{seed}-{kind}.pt")
            actual = transfer.predict(model, transfer.values(memo, "calibration", kind, seed), rows["calibration"], deadline)
            same_prediction(actual, cal["prediction"]); replays += 1
            threshold = transfer.decision_threshold(cal["prediction"].numpy(), rows["calibration"]["labels"].numpy())
            calibration = {"seed": seed, "kind": kind, "threshold": threshold,
                           "metrics": transfer.metrics(cal["prediction"], rows["calibration"], threshold),
                           "parameters": sum(p.numel() for p in model.parameters())}
            if cal["record"] != calibration or threshold != lock["thresholds"][f"{seed}-{kind}"]:
                raise ValueError("threshold calibration mismatch")
            calibrations.append(calibration)
            for split in ("test", "ood"):
                grids = transfer.values(memo, split, kind, seed); variants = {kind: grids}
                if kind == "compact":
                    permutation = torch.randperm(len(grids), generator=torch.Generator().manual_seed(seed + 30000))
                    variants.update(shuffled=grids[permutation], zeroed=torch.zeros_like(grids))
                for variant, value in variants.items():
                    saved = load(output, report, f"test-{seed}-{variant}-{split}.pt")
                    actual = transfer.predict(model, value, rows[split], deadline)
                    same_prediction(actual, saved["prediction"]); replays += 1
                    record = {"seed": seed, "kind": variant, "split": split, "threshold": threshold,
                              "metrics": transfer.metrics(saved["prediction"], rows[split], threshold)}
                    if record != saved["record"]:
                        raise ValueError("collision metrics mismatch")
                    results.append(record); predictions[seed, variant, split] = saved["prediction"]
            del model
        del memo
        print(json.dumps({"transfer_seed_verified": seed}), flush=True)
    comparisons = []
    for seed in transfer.PLAN["seeds"]:
        for split in ("test", "ood"):
            for kind in ("raw", "spatial", "null", "shuffled"):
                row = rows[split]
                comparisons.append({"seed": seed, "split": split, "reference": kind, "negative_log_loss_difference":
                    common.bootstrap(transfer.per_geometry_score(predictions[seed, "compact", split], row),
                                     transfer.per_geometry_score(predictions[seed, kind, split], row),
                                     row["ids"], row["families"], row["densities"])})
    if results != report["records"] or calibrations != report["calibration"] or comparisons != report["comparisons"]:
        raise ValueError("transfer aggregate mismatch")
    if transfer.engineering_screen(results) != report["engineering_screen"]:
        raise ValueError("engineering assessment mismatch")
    for split in ("test", "ood"):
        row = rows[split]; definite = row["visible_hit"] | ~row["unknown_path"]
        expected = {"coverage": float(definite.float().mean()), "abstain_fraction": float((~definite).float().mean()),
                    "definite_errors": int(((row["visible_hit"] != row["labels"]) & definite).sum()), "definite_count": int(definite.sum())}
        if report["visible_grid_rule"][split] != expected:
            raise ValueError("visible rule mismatch")
    return {"prediction_groups_replayed": replays, "encoder_batches_replayed": batches,
            "heads_verified": 12, "queries_replayed": queries_checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["robustness", "transfer"])
    for name in ("registration", "data-root", "package", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    env, rows = common.registration(args.registration, args.kind, args.data_root)
    report = checked_report(args.output, env)
    common.replication.configure_cuda(); deadline = time.monotonic() + 1800
    verify = verify_robustness if args.kind == "robustness" else verify_transfer
    evidence = verify(report, rows, args.output, args.package, deadline)
    evidence.update(report_payload_sha256=report["report_payload_sha256"], verified=True,
                    artifacts_verified=len(report["artifacts"]), elapsed_seconds=time.monotonic()-start)
    common.base.write_json(args.output / "verification.json", evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
