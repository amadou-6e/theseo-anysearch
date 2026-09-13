"""Export and verify the three experimentally validated fine729 artifacts."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch

from theseo_anysearch.garden import compact_package as package
from theseo_anysearch.garden.pilots import compact_replication as study

RUN_ID = "compact-package-v1-run1"
CAP = 1800


def execute(run, source, output, spec, ledger_path):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if study.d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden", "scripts/package_compact_encoder.py"):
        raise ValueError("commit package source first")
    with study.search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = study.search.reserve_budget(ledger_path, RUN_ID, CAP)
        if ledger["runs"][RUN_ID]["settled"] or (output / "manifest.json").exists():
            raise ValueError("package run already attempted; do not overwrite")
        try:
            cutoff = start + CAP - 300
            output.mkdir(parents=True, exist_ok=True)
            identity = {"run_id": RUN_ID, "spec_commit": spec, "source_commit": study.d.base.git("rev-parse", "HEAD"),
                        "source_report": study.PLAN["source_report"], "contract": package.CONTRACT,
                        "validation": "first_and_last_32_development_rows_each_seed", "cap_seconds": CAP}
            # The source of this export is the replication, not its search predecessor.
            report = json.loads((run / "report.json").read_text()); digest = report.pop("report_payload_sha256")
            if digest != "f7d65863a81c3c905d24649d78a4d2d55ce747ccc0c89d361feb796d4bfc7c10" or package.payload_sha256(report) != digest:
                raise ValueError("replication report mismatch")
            if study.assess(report["development"]) != report["assessment"] or not report["assessment"]["experimental_packaging_ready"]:
                raise ValueError("replication not ready")
            identity["source_report"] = digest
            study.search.atomic_json(output / "preregistration.json", identity)
            payload = study.check_registration(report["registration"])
            if payload["source_encoders"] != study.corpus.old.helpers.prior.sources(source):
                raise ValueError("backbone provenance mismatch")
            for name, expected in report["artifacts"].items():
                if Path(name).name != name or package.file_sha(run / name) != expected:
                    raise ValueError("source artifact mismatch")
                study.d.check_deadline(cutoff)

            def load(name):
                if name not in report["artifacts"]:
                    raise ValueError("unhashed source")
                return torch.load(run / name, map_location="cpu", weights_only=True)

            def stage(name, model):
                state = report["stages"][name]; saved = load(state["checkpoint"])
                if saved["state"] != state or state["steps"] != 8192:
                    raise ValueError("source stage mismatch")
                model.load_state_dict(saved["model"])
                if package.encoder_state_sha256(model) != state["model_state_sha256"]:
                    raise ValueError("source state mismatch")
                return model

            manifest = {**identity, "artifacts": {}, "limits": ["fixed_backbone", "synthetic_local_geometry",
                         "native_head_only", "narrow_random_field_f1_margin", "not_navigation_or_topology_validated"]}
            for record in report["records"]:
                seed = record["seed"]
                model = stage(f"train-{seed}", study.make_model(seed))
                if package.encoder_state_sha256(model.aggregation) != record["aggregation_state_sha256"]:
                    raise ValueError("source aggregation mismatch")
                artifact = load(f"assessment-{seed}.pt")
                if artifact["record"] != record:
                    raise ValueError("source assessment mismatch")
                backbone = package.empty_encoder().backbone
                backbone.load_state_dict(torch.load(source / payload["source_encoders"][0]["path"], map_location="cpu", weights_only=True))
                if package.encoder_state_sha256(backbone) != payload["source_encoders"][0]["state_hash"]:
                    raise ValueError("source backbone mismatch")
                stats = artifact["statistics"]
                encoder = package.FrozenCompactEncoder(backbone, model.aggregation, stats["mean"], stats["scale"])
                head = stage(f"probe-{seed}", study.make_head(seed))
                name = f"seed{seed}.pt"; temp = output / f"seed{seed}.partial"
                torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, temp)
                os.replace(temp, output / name)
                manifest["artifacts"][str(seed)] = {"file": name, "sha256": package.file_sha(output / name),
                    "encoder_state_sha256": package.encoder_state_sha256(encoder), "head_state_sha256": package.encoder_state_sha256(head),
                    "thresholds": record["native"]["thresholds"], "source_aggregation_state_sha256": record["aggregation_state_sha256"],
                    "source_train_checkpoint_sha256": report["artifacts"][report["stages"][f"train-{seed}"]["checkpoint"]],
                    "source_head_checkpoint_sha256": report["artifacts"][report["stages"][f"probe-{seed}"]["checkpoint"]]}
            manifest["manifest_payload_sha256"] = package.payload_sha256(manifest)
            study.search.atomic_json(output / "manifest.json", manifest)
            study.configure_cuda()
            rows = study.data(["development"])
            if study.corpus.old.identity(rows) != {"development": payload["data"]["development"]}:
                raise ValueError("development identity mismatch")
            row = rows["development"]; checks = []
            for seed in study.SEEDS:
                encoder, head, _ = package.load_compact_package(output, seed=seed, device="cuda")
                expected = load(f"development-{seed}.pt")["predictions"]["native"]
                for first in (0, len(row["targets"]) - 32):
                    occupancy = row["occupancy"][first:first+32].cuda()
                    unknown = row["hidden"][first:first+32].cuda()
                    code = encoder(occupancy, unknown)
                    prediction = study.prior.predict(head, "vector", code.cpu(), cutoff)
                    if not torch.allclose(prediction, expected[first:first+32], rtol=1e-5, atol=1e-6):
                        raise ValueError("direct package prediction mismatch")
                original_hash = package.encoder_state_sha256(encoder)
                disposable = copy.deepcopy(head).train().requires_grad_(True)
                old_head = package.encoder_state_sha256(disposable)
                optimizer = torch.optim.AdamW(disposable.parameters(), lr=.001)
                code = encoder(row["occupancy"][:2].cuda(), row["hidden"][:2].cuda())
                indices = torch.arange(256, device="cuda")[None].expand(2, -1)
                disposable(code, indices).square().mean().backward()
                gradients = [p.grad for p in disposable.parameters() if p.grad is not None]
                if not gradients or not all(g.isfinite().all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
                    raise ValueError("invalid head gradients")
                optimizer.step()
                if old_head == package.encoder_state_sha256(disposable) or original_hash != package.encoder_state_sha256(encoder) or any(p.grad is not None for p in encoder.parameters()):
                    raise ValueError("head-only training contract failed")
                checks.append({"seed": seed, "direct_rows_verified": 64, "head_only_step": "passed"})
                del encoder, head, disposable, optimizer
                study.d.check_deadline(cutoff)
            encoder, _, _ = package.load_compact_package(output, device="cuda")
            occupancy = row["occupancy"][:1].cuda(); unknown = row["hidden"][:1].cuda()
            for _ in range(10): encoder(occupancy, unknown)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(50):
                t = time.perf_counter(); encoder(occupancy, unknown); torch.cuda.synchronize()
                timings.append((time.perf_counter() - t) * 1000)
            study.d.check_deadline(cutoff)
            evidence = {"identity": identity, "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                        "checks": checks, "batch1_latency_ms": {"median": float(torch.tensor(timings).median()),
                          "p95": float(torch.tensor(timings).quantile(.95))}, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "elapsed_seconds": time.monotonic() - start + 300, "status": "experimental_package_verified",
                        "promotion_eligible": False}
            evidence["report_payload_sha256"] = package.payload_sha256(evidence)
            study.search.atomic_json(output / "report.json", evidence)
            ledger["runs"][RUN_ID].update(charged_seconds=time.monotonic() - start + 300, settled=True)
            study.search.atomic_json(ledger_path, ledger)
            print(json.dumps(evidence, indent=2), flush=True)
        except BaseException:
            ledger["runs"][RUN_ID].update(charged_seconds=CAP, settled=True, failed=True)
            study.search.atomic_json(ledger_path, ledger)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--spec-sha", required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    args = parser.parse_args()
    execute(args.run, args.source, args.output, args.spec_sha, args.budget_ledger)
