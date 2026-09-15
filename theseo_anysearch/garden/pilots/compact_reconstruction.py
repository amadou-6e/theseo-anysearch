"""Bounded small-field aggregation training with independent frozen readouts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .. import compact_reconstruction as models
from .. import compact_readout as heads
from . import compact_diversity as previous
from . import compact_diversity_data as corpus
from . import compact_diagnosis as diagnosis
from . import compact_nonlinear as nonlinear
from . import compact_search as search

d = corpus.d
COUNTS = {"train": 1536, "probe": 768, "selection": 192, "development": 192}
PLAN = {"run_id": "compact-reconstruction-v1-run1", "cap_seconds": 14400,
        "fit_cap_seconds": 1800, "preparation_cap_seconds": 3600,
        "overhead_reserve_seconds": 300, "batch": 32, "queries": 256,
        "steps": 4096, "counts": COUNTS,
        "source_report": "39503c4fdbb641581a3d1d587827e6fe30e46126dae96e46f9a191d9092fd6e8"}


def data(splits=None):
    return corpus.data(splits, study_id="compact-reconstruction-v1", counts=COUNTS, bank_splits=("train", "probe"))


def configs():
    return [{"id": i, "kind": k, "dimension": n, "decoder": h}
            for i, (k, n, h) in enumerate((k, n, h) for k in ("grid", "residual")
                                         for n in (64, 128) for h in ("linear", "conv"))]


def check_registration(env):
    payload = env["payload"]
    if d.base.payload_sha256(payload) != env["identity_sha256"] or payload["plan"] != PLAN or payload["configs"] != configs():
        raise ValueError("registration mismatch")
    return payload


def freeze(path, spec, report_paths):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    reports = []
    for path_in in report_paths:
        report = json.loads(path_in.read_text()); digest = report.pop("report_payload_sha256")
        if digest != d.base.payload_sha256(report):
            raise ValueError("prior report mismatch")
        reports.append((report, digest))
    if len(reports) != 3 or reports[-1][1] != PLAN["source_report"]:
        raise ValueError("require search/diversity/readout reports in order")
    rows = data(); corpus.support(rows)
    parents = {h for r, _ in reports for split in r["registration"]["payload"]["data"].values() for h in split["parents"]}
    if any(h in parents for row in rows.values() for h in row["parents"]):
        raise ValueError("reused parent")
    payload = {"plan": PLAN, "configs": configs(), "source_commit": d.base.git("rev-parse", "HEAD"),
               "spec_commit": spec, "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-reconstruction.md",
               "prior_reports": [digest for _, digest in reports], "data": corpus.old.identity(rows),
               "source_encoders": reports[-1][0]["registration"]["payload"]["source_encoders"],
               "preparation_seconds": time.monotonic() - start}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def cache_shape(row):
    return (len(row["hidden"].reshape(-1, 33, 33, 33)), 8, 33, 33, 33)


def cache_bytes(shape):
    if len(shape) != 5 or shape[0] < 1 or tuple(shape[1:]) != (8, 33, 33, 33):
        raise ValueError("invalid full-feature cache shape")
    return int(np.prod(shape)) * 4


def open_cache(path, shape, digest):
    if path.stat().st_size != cache_bytes(shape) or previous.file_hash(path) != digest:
        raise ValueError("feature cache size/hash mismatch")
    return np.memmap(path, mode="r", dtype="<f4", shape=tuple(shape))


@torch.no_grad()
def create_cache(path, backbone, row, deadline):
    shape = cache_shape(row); partial = path.with_suffix(".partial")
    if path.exists() or partial.exists():
        raise ValueError("unregistered cache exists; audit before reuse")
    if shutil.disk_usage(path.parent).free < cache_bytes(shape) + 20 * 1024**3:
        raise ValueError("insufficient disk headroom")
    state_hash = d.base.encoder_state_sha256(backbone)
    mapped = np.memmap(partial, mode="w+", dtype="<f4", shape=shape)
    hidden = row["hidden"].reshape(-1, 33, 33, 33); bank = len(hidden) // len(row["targets"])
    try:
        for i in range(0, len(hidden), 16):
            d.check_deadline(deadline)
            ids = torch.arange(i, min(i + 16, len(hidden))) // bank
            occ = row["occupancy"][ids].cuda().float(); mask = hidden[i:i + 16, None].cuda()
            volume = backbone(d.base.VoxelLevel.from_occupancy(occ, unknown_mask=mask), mask).local_feature_volume
            mapped[i:i + len(ids)] = volume.cpu().numpy()
            if i % 1024 == 0:
                print(json.dumps({"cache": path.name, "entries": i, "total": len(hidden)}), flush=True)
        mapped.flush()
    finally:
        mapped._mmap.close()
    if d.base.encoder_state_sha256(backbone) != state_hash:
        raise ValueError("cache extraction mutated backbone")
    os.replace(partial, path)
    return {"shape": list(shape), "sha256": previous.file_hash(path), "backbone_hash": state_hash}


def fit_labels(row):
    bank = row["hidden"].shape[1] if row["hidden"].ndim == 5 else 1
    return row["targets"], corpus.old.helpers.prior.crop(row["hidden"], 17).reshape(-1, 4913), bank


def batch_labels(labels, ids, indices):
    target, hidden, bank = labels
    ids, indices = ids.to(target.device), indices.to(target.device)
    return (target[ids // bank].gather(2, indices[:, None].expand(-1, 3, -1)),
            hidden[ids].gather(1, indices))


def model_input(kind, values, ids, indices):
    if kind == "spatial":
        return torch.from_numpy(models.spatial_queries(values, ids.numpy(), indices.numpy())).cuda()
    if kind == "volume":
        return torch.from_numpy(np.array(values[ids.numpy()], copy=True)).cuda()
    return values[ids].cuda()


@torch.no_grad()
def predict(model, kind, values, deadline):
    mode = model.training; model.eval(); parts = []
    for i in range(0, len(values), 8):
        d.check_deadline(deadline)
        ids = torch.arange(i, min(i + 8, len(values)))
        indices = torch.arange(4913)[None].expand(len(ids), -1)
        logits = model(model_input(kind, values, ids, indices), indices.cuda())
        parts.append(torch.cat((logits[:, :2].sigmoid(), logits[:, 2:].clamp(0, 1)), 1).cpu())
    model.train(mode)
    return torch.cat(parts)


@torch.no_grad()
def vectors(aggregation, values, deadline):
    aggregation.eval(); state_hash = d.base.encoder_state_sha256(aggregation); parts = []
    for i in range(0, len(values), 32):
        d.check_deadline(deadline)
        parts.append(aggregation(torch.from_numpy(np.array(values[i:i + 32], copy=True)).cuda()).cpu())
    if d.base.encoder_state_sha256(aggregation) != state_hash:
        raise ValueError("extraction mutated aggregation")
    return torch.cat(parts)


def resume_charge(progress):
    return progress["elapsed_seconds"] + (progress["inflight"]["cap_seconds"] if progress["inflight"] else 0)


class Campaign:
    def __init__(self, output, env, progress, start, elapsed_before, plan=None):
        self.plan = PLAN if plan is None else plan
        self.output, self.env, self.progress = output, env, progress
        self.start, self.elapsed_before = start, elapsed_before
        self.deadline = start + self.plan["cap_seconds"] - elapsed_before

    def save(self):
        self.progress["elapsed_seconds"] = self.elapsed_before + time.monotonic() - self.start
        search.atomic_json(self.output / "progress.json", self.progress)

    def phase(self, name, cap):
        d.check_deadline(self.deadline)
        self.progress["inflight"] = {"name": name, "cap_seconds": cap}; self.save()
        return min(self.deadline, time.monotonic() + cap)

    def artifact(self, name, value):
        path = self.output / name; temporary = path.with_suffix(".partial")
        torch.save(value, temporary); os.replace(temporary, path)
        self.progress["artifacts"][name] = previous.file_hash(path); self.save()

    def load(self, name):
        if previous.file_hash(self.output / name) != self.progress["artifacts"].get(name):
            raise ValueError("unverified checkpoint")
        return torch.load(self.output / name, map_location="cpu", weights_only=True)

    def cache(self, split, backbone, row, cutoff):
        name = f"{split}-features.f32"; key = corpus.old.identity({split: row})
        if name in self.progress["caches"]:
            record = self.progress["caches"][name]
            if record["data"] != key or record["backbone_hash"] != d.base.encoder_state_sha256(backbone):
                raise ValueError("cache identity mismatch")
        else:
            record = create_cache(self.output / name, backbone, row, cutoff)
            record["data"] = key
            self.progress["caches"][name] = record
            self.progress["artifacts"][name] = record["sha256"]; self.save()
        return open_cache(self.output / name, record["shape"], record["sha256"])

    def fit(self, name, model, kind, values, row, lr=.001):
        state = self.progress["stages"].get(name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        rng = torch.Generator().manual_seed(self.plan.get("sampling_seed", 39700)); first = 0; prior = 0.; curve = []
        if state:
            saved = self.load(state["checkpoint"])
            if saved["state"] != state:
                raise ValueError("stage state mismatch")
            model.load_state_dict(saved["model"])
            if d.base.encoder_state_sha256(model) != state["model_state_sha256"]:
                raise ValueError("saved model mismatch")
            if state["steps"] == self.plan["steps"]:
                return model
            optimizer.load_state_dict(saved["optimizer"]); rng.set_state(saved["rng"])
            first, prior, curve = state["steps"], state["fit_seconds"], state["curve"][:]
        cutoff = self.phase(name, self.plan["fit_cap_seconds"])
        fit_start = time.monotonic(); cutoff = min(cutoff, fit_start + self.plan["fit_cap_seconds"] - prior)
        prevalence = row["targets"][:, :2].mean((0, 2)).cuda()
        weights = (1 - prevalence) / prevalence.clamp_min(1e-5)
        target, hidden, bank = fit_labels(row)
        labels = (target.cuda(), hidden.cuda(), bank)
        torch.cuda.reset_peak_memory_stats(); model.train()
        base_lr = lr
        for step in range(first + 1, self.plan["steps"] + 1):
            d.check_deadline(cutoff)
            lr = nonlinear.learning_rate({"steps": self.plan["steps"], "lr": base_lr}, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            ids = torch.randint(len(values), (self.plan["batch"],), generator=rng)
            indices = torch.randint(4913, (self.plan["batch"], self.plan["queries"]), generator=rng)
            logits = model(model_input(kind, values, ids, indices), indices.cuda())
            target, hidden = batch_labels(labels, ids, indices)
            loss = search.objective(logits, target, hidden, weights, {"boundary_weight": 1., "distance_weight": 10.})
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite reconstruction loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
            if step % 512 == 0:
                curve.append({"step": step, "minibatch_loss": float(loss.detach()), "lr": lr})
                print(json.dumps({"stage": name, **curve[-1]}), flush=True)
            if step % 1024 == 0:
                checkpoint = f"{name}-step-{step}.pt"
                state = {"steps": step, "checkpoint": checkpoint, "curve": curve[:],
                         "model_state_sha256": d.base.encoder_state_sha256(model),
                         "fit_seconds": prior + time.monotonic() - fit_start,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "sampled_queries": step * self.plan["batch"] * self.plan["queries"],
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                self.artifact(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                           "rng": rng.get_state(), "state": state})
                self.progress["stages"][name] = state; self.save()
        self.progress["inflight"] = None; self.save()
        return model


def metrics(prediction, row):
    thresholds = corpus.old.select_thresholds(prediction, row)
    return {"selection": diagnosis.compact_metrics(prediction, row, thresholds), "thresholds": thresholds}


def make_model(config):
    torch.manual_seed(397)
    return models.ReconstructionModel(config).cuda()


def run(path, source, output, ledger_path):
    start = time.monotonic(); env = json.loads(path.read_text()); payload = check_registration(env)
    if d.base.git("diff", payload["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    if payload["source_encoders"] != corpus.old.helpers.prior.sources(source):
        raise ValueError("initializer mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = search.reserve_budget(ledger_path, PLAN["run_id"], PLAN["cap_seconds"])
        if (output / "report.json").exists():
            done = json.loads((output / "report.json").read_text()); digest = done.pop("report_payload_sha256")
            if d.base.payload_sha256(done) != digest or done["registration"] != env:
                raise ValueError("completed report mismatch")
            print("Completed; no fits restarted.", flush=True); return
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "identity_sha256": env["identity_sha256"], "elapsed_seconds": payload["preparation_seconds"] + PLAN["overhead_reserve_seconds"],
            "inflight": None, "stages": {}, "records": [], "controls": {}, "caches": {}, "artifacts": {}}
        if progress["identity_sha256"] != env["identity_sha256"]:
            raise ValueError("progress identity mismatch")
        campaign = Campaign(output, env, progress, start, resume_charge(progress))
        cutoff = campaign.phase("preparation", PLAN["preparation_cap_seconds"])
        for name, digest in progress["artifacts"].items():
            if previous.file_hash(output / name) != digest:
                raise ValueError("artifact mismatch")
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        rows = data(["train", "probe", "selection"]); corpus.support(rows)
        if corpus.old.identity(rows) != {s: payload["data"][s] for s in rows}:
            raise ValueError("data mismatch")
        backbone = d.base.make_encoder(0, torch.device("cuda"))
        backbone.load_state_dict(torch.load(source / payload["source_encoders"][0]["path"], map_location="cuda", weights_only=True))
        backbone.eval().requires_grad_(False)
        if d.base.encoder_state_sha256(backbone) != payload["source_encoders"][0]["state_hash"]:
            raise ValueError("backbone hash mismatch")
        cache = {s: campaign.cache(s, backbone, rows[s], cutoff) for s in rows}
        progress["inflight"] = None; campaign.save()
        torch.manual_seed(397)
        spatial = campaign.fit("spatial", models.SpatialReference().cuda(), "spatial", cache["probe"], rows["probe"])
        if "spatial" not in progress["controls"]:
            cutoff = campaign.phase("spatial-assessment", PLAN["fit_cap_seconds"])
            prediction = predict(spatial, "spatial", cache["selection"], cutoff)
            progress["controls"]["spatial"] = metrics(prediction, rows["selection"])
            campaign.artifact("spatial-selection.pt", prediction)
            print(json.dumps({"spatial_reference": progress["controls"]["spatial"]}), flush=True)
        del spatial
        for config in configs():
            if any(r["config"] == config for r in progress["records"]):
                continue
            model = campaign.fit(f"train-{config['id']}", make_model(config), "volume", cache["train"], rows["train"])
            cutoff = campaign.phase(f"extract-{config['id']}", PLAN["fit_cap_seconds"])
            aggregation = model.aggregation.eval().requires_grad_(False)
            aggregation_hash = d.base.encoder_state_sha256(aggregation)
            vector_path = f"vectors-{config['id']}.pt"
            if vector_path in progress["artifacts"]:
                stored = campaign.load(vector_path)
                if stored["aggregation_hash"] != aggregation_hash:
                    raise ValueError("vector cache aggregation mismatch")
            else:
                stored = {"aggregation_hash": aggregation_hash,
                          "probe": vectors(aggregation, cache["probe"], cutoff),
                          "selection": vectors(aggregation, cache["selection"], cutoff)}
                campaign.artifact(vector_path, stored)
            z, stats = heads.normalized_vectors(stored["probe"])
            selected_z = heads.normalize(stored["selection"], stats)
            trained_prediction = predict(model, "volume", cache["selection"], cutoff)
            torch.manual_seed(397)
            head = campaign.fit(f"probe-{config['id']}", heads.ConvolutionalReadout(config["dimension"]).cuda(), "vector", z, rows["probe"])
            cutoff = campaign.phase(f"assess-{config['id']}", PLAN["fit_cap_seconds"])
            prediction = predict(head, "vector", selected_z, cutoff)
            result = metrics(prediction, rows["selection"])
            permutation = torch.randperm(len(selected_z), generator=torch.Generator().manual_seed(397))
            shuffled = predict(head, "vector", selected_z[permutation], cutoff)
            if d.base.encoder_state_sha256(aggregation) != aggregation_hash:
                raise ValueError("independent fit mutated encoder")
            singular = torch.linalg.svdvals(stored["probe"].double() - stored["probe"].double().mean(0))
            probability = singular.square() / singular.square().sum().clamp_min(1e-30)
            rank_entropy = float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))
            record = {"config": config, "steps": PLAN["steps"], "aggregation_state_sha256": aggregation_hash,
                      "aggregation_parameters": sum(p.numel() for p in aggregation.parameters()),
                      "variance_entropy_rank": rank_entropy, "trained_decoder": metrics(trained_prediction, rows["selection"]),
                      "shuffled": diagnosis.compact_metrics(shuffled, rows["selection"], result["thresholds"]), **result}
            campaign.artifact(f"assessment-{config['id']}.pt", {"record": record, "statistics": stats,
                              "prediction": prediction, "shuffled": shuffled, "trained_prediction": trained_prediction})
            progress["records"].append(record); progress["inflight"] = None; campaign.save()
            print(json.dumps({"candidate": config, "scores": {k: v["score"] for k, v in result["selection"].items()}}), flush=True)
            del model, aggregation, head, stored, z, selected_z
        torch.manual_seed(397)
        zeros = torch.zeros(len(cache["probe"]), 64)
        null = campaign.fit("null", heads.ConvolutionalReadout().cuda(), "vector", zeros, rows["probe"])
        cutoff = campaign.phase("final-assessment", PLAN["preparation_cap_seconds"])
        if "null" not in progress["controls"]:
            prediction = predict(null, "vector", torch.zeros(len(cache["selection"]), 64), cutoff)
            progress["controls"]["null"] = metrics(prediction, rows["selection"])
            campaign.artifact("null-selection.pt", prediction)
        del null
        finalists = sorted(progress["records"], key=corpus.rank, reverse=True)[:2]
        lock = {"identity_sha256": env["identity_sha256"], "ids": [r["config"]["id"] for r in finalists]}
        lock_path = output / "selection-lock.json"
        if lock_path.exists():
            if json.loads(lock_path.read_text()) != lock:
                raise ValueError("selection lock changed")
        else:
            d.base.write_json(lock_path, lock)
        development = data(["development"])
        if corpus.old.identity(development) != {"development": payload["data"]["development"]}:
            raise ValueError("development identity mismatch")
        dev_cache = campaign.cache("development", backbone, development["development"], cutoff)
        assessments = []
        for record in finalists:
            config = record["config"]; model = make_model(config)
            model.load_state_dict(campaign.load(progress["stages"][f"train-{config['id']}"]["checkpoint"])["model"])
            assessment = campaign.load(f"assessment-{config['id']}.pt")
            head = heads.ConvolutionalReadout(config["dimension"]).cuda()
            head.load_state_dict(campaign.load(progress["stages"][f"probe-{config['id']}"]["checkpoint"])["model"])
            z = heads.normalize(vectors(model.aggregation, dev_cache, cutoff), assessment["statistics"])
            prediction = predict(head, "vector", z, cutoff)
            assessments.append({"id": config["id"], "metrics": diagnosis.compact_metrics(prediction, development["development"], record["thresholds"])})
            campaign.artifact(f"development-{config['id']}.pt", prediction)
        spatial = models.SpatialReference().cuda()
        spatial.load_state_dict(campaign.load(progress["stages"]["spatial"]["checkpoint"])["model"])
        prediction = predict(spatial, "spatial", dev_cache, cutoff)
        campaign.artifact("spatial-development.pt", prediction)
        spatial_dev = diagnosis.compact_metrics(prediction, development["development"], progress["controls"]["spatial"]["thresholds"])
        d.check_deadline(cutoff)
        evidence = {"registration": env, "status": "completed", "records": progress["records"], "stages": progress["stages"],
                    "controls": progress["controls"], "selected_id": lock["ids"][0], "development": assessments,
                    "spatial_development": spatial_dev, "artifacts": progress["artifacts"], "caches": progress["caches"],
                    "elapsed_seconds": campaign.elapsed_before + time.monotonic() - start, "promotion_eligible": False,
                    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}}
        evidence["report_payload_sha256"] = d.base.payload_sha256(evidence); d.base.write_json(output / "report.json", evidence)
        ledger["runs"][PLAN["run_id"]].update(charged_seconds=campaign.elapsed_before + time.monotonic() - start, settled=True)
        search.atomic_json(ledger_path, ledger)
        progress["inflight"] = None; campaign.save()
        print(json.dumps({"completed": True, "selected_id": lock["ids"][0], "development": assessments}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--spec-sha")
    parser.add_argument("--prior-reports", type=Path, nargs=3); parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.prior_reports)
    else:
        run(args.registration, args.source, args.output, args.budget_ledger)


if __name__ == "__main__":
    main()
