"""Oracle-code, learned-table and voxel-encoder tiny-corpus learnability checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..compact import CompactEncoder, query_coordinates
from . import compact_controls as previous

d = previous.d
PLAN = {"study_id": "compact-learnability-v1", "sources": ["oracle", "table", "encoder"],
        "decoders": ["plain32", "fourier128"], "steps": 2048, "queries": 256,
        "checkpoints": [128, 512, 1024, 2048], "cap_seconds": 1800, "fit_cap_seconds": 600,
        "overfit_iou": .95, "overfit_f1": .97, "generalization_claim": False}


def analytic(code, xyz):
    phase = 8 * code[:, None, 3] * (xyz * code[:, None, :3]).sum(-1) + np.pi * code[:, None, 4]
    return phase.sin().abs() <= code[:, None, 5]


def data():
    occs, masks, codes, targets, ids, hashes = [], [], [], [], [], []
    coords = np.stack(np.meshgrid(*([np.arange(-24, 25)/8]*3), indexing="ij"))
    for i, fraction in enumerate((.08, .16, .28, .16)):
        gid = f"compact-learnability-v1-scene-{i}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(gid.encode()).digest()[:8], "little"))
        normal = rng.normal(size=3); normal /= np.linalg.norm(normal)
        frequency = rng.uniform(4, 8); phase = rng.uniform(-np.pi, np.pi)
        field = np.abs(np.sin(frequency*(coords*normal[:, None, None, None]).sum(0)+phase))
        threshold = np.quantile(field, fraction, method="midpoint")
        occ = field <= threshold
        maskrng = np.random.default_rng(int.from_bytes(hashlib.sha256((gid+"-mask").encode()).digest()[:8], "little"))
        occs.append(previous.previous.prior.crop(occ, 33).copy())
        masks.append(previous.previous.prior.crop(maskrng.random(occ.shape)<.2, 33).copy())
        targets.append(previous.previous.prior.crop(occ, 17).copy())
        codes.append([*normal, frequency/8, phase/np.pi, threshold])
        ids.append(gid); hashes.append(hashlib.sha256(occ.tobytes()).hexdigest())
    rows = {"occupancy": torch.tensor(np.stack(occs), dtype=torch.float32),
            "hidden": torch.tensor(np.stack(masks), dtype=torch.bool),
            "targets": torch.tensor(np.stack(targets), dtype=torch.float32).flatten(1),
            "codes": torch.tensor(codes, dtype=torch.float64), "ids": ids, "parents": hashes}
    xyz = query_coordinates(torch.arange(4913).expand(4, -1)).double()
    if not torch.equal(analytic(rows["codes"], xyz), rows["targets"].bool()):
        raise ValueError("analytic code and generated targets disagree")
    if len(set(hashes)) != 4: raise ValueError("duplicate parent")
    return rows


def identity(rows):
    return {"ids": rows["ids"], "parents": rows["parents"],
            "arrays": {k: d.digest_tensor(rows[k]) for k in ("occupancy", "hidden", "targets", "codes")}}


def coordinates(indices, bundle):
    xyz = query_coordinates(indices)
    if bundle == "plain32": return xyz
    if bundle != "fourier128": raise ValueError("unknown decoder bundle")
    frequencies = xyz.new_tensor([1, 2, 4, 8])
    phase = np.pi * xyz[..., None] * frequencies
    return torch.cat((xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


class QueryDecoder(nn.Module):
    def __init__(self, dimension, bundle):
        super().__init__(); self.bundle = bundle
        if bundle == "plain32":
            self.net = nn.Sequential(nn.Linear(dimension+3, 32), nn.SiLU(), nn.Linear(32, 1))
        elif bundle == "fourier128":
            self.net = nn.Sequential(nn.Linear(dimension+27, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
        else: raise ValueError("unknown decoder bundle")

    def forward(self, code, indices):
        x = torch.cat((code[:, None].expand(-1, indices.shape[1], -1), coordinates(indices, self.bundle)), -1)
        return self.net(x).squeeze(-1)


def freeze(path, spec, source):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"): raise ValueError("commit source first")
    p = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
         "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-learnability.md",
         "source_encoders": previous.previous.prior.sources(source), "data": identity(data()),
         "preparation_seconds": time.monotonic()-start}
    d.base.write_json(path, {"payload": p, "identity_sha256": d.base.payload_sha256(p)})


def code_for(source_kind, provider, level, hidden, oracle):
    if source_kind == "oracle": return oracle
    if source_kind == "table": return provider(torch.arange(4, device=oracle.device))
    return provider(level, hidden)


@torch.no_grad()
def evaluate(provider, decoder, kind, level, hidden, oracle, target):
    code = code_for(kind, provider, level, hidden, oracle)
    predictions = torch.cat([decoder(code, torch.arange(i, min(i+256, 4913), device=code.device).expand(4, -1)).sigmoid() for i in range(0, 4913, 256)], 1)
    p, y = predictions.cpu().numpy(), target.cpu().numpy()
    iou = previous.metrics(p, y, .5, "occupied_iou")
    f1 = previous.metrics(p, y, .5, "boundary_f1")["fixed05_score"]
    return predictions, {"iou": iou["fixed05_score"], "f1": f1, "auprc": iou["auprc"],
                         "bce": iou["log_loss_nats"], "positive_rate": iou["positive_rate05"],
                         "probability_range": [float(p.min()), float(p.max())],
                         "per_scene": [{"iou": previous.metrics(p[i:i+1], y[i:i+1], .5, "occupied_iou")["fixed05_score"],
                                        "f1": previous.metrics(p[i:i+1], y[i:i+1], .5, "boundary_f1")["fixed05_score"]} for i in range(4)]}


def run(path, source, output):
    start = time.monotonic(); env = json.loads(path.read_text()); p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or previous.previous.prior.sources(source) != p["source_encoders"]: raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"): raise ValueError("source mismatch")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    rows = data()
    if identity(rows) != p["data"]: raise ValueError("data mismatch")
    output.mkdir(parents=True, exist_ok=False)
    hidden = rows["hidden"].cuda()[:, None]
    level = d.base.VoxelLevel.from_occupancy(rows["occupancy"].cuda(), unknown_mask=hidden)
    oracle = rows["codes"].float().cuda(); target = rows["targets"].cuda()
    positive_weight = (1-target).sum()/target.sum()
    deadline = start + PLAN["cap_seconds"] - p["preparation_seconds"]
    records = []
    for kind in PLAN["sources"]:
        for bundle in PLAN["decoders"]:
            torch.manual_seed(385)
            if kind == "encoder":
                backbone = d.base.make_encoder(0, torch.device("cuda"))
                backbone.load_state_dict(torch.load(source/p["source_encoders"][0]["path"], map_location="cuda", weights_only=True))
                if d.base.encoder_state_sha256(backbone) != p["source_encoders"][0]["state_hash"]: raise ValueError("backbone mismatch")
                torch.manual_seed(385)
                provider = CompactEncoder(backbone, 128, "grid", joint=True).cuda()
            elif kind == "table": provider = nn.Embedding(4, 128).cuda()
            else: provider = None
            decoder = QueryDecoder(6 if kind == "oracle" else 128, bundle).cuda()
            parameters = list(decoder.parameters()) + ([x for x in provider.parameters() if x.requires_grad] if provider is not None else [])
            opt = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
            rng = torch.Generator(device="cuda").manual_seed(38500)
            fit_start = time.monotonic(); fit_deadline = min(deadline, fit_start+PLAN["fit_cap_seconds"])
            torch.cuda.reset_peak_memory_stats(); curve = []
            for step in range(PLAN["steps"]):
                d.check_deadline(fit_deadline)
                idx = torch.randint(4913, (4, 256), generator=rng, device="cuda")
                code = code_for(kind, provider, level, hidden, oracle)
                logits = decoder(code, idx)
                loss = F.binary_cross_entropy_with_logits(logits, target.gather(1, idx), pos_weight=positive_weight)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if step+1 in PLAN["checkpoints"]:
                    pred, scores = evaluate(provider, decoder, kind, level, hidden, oracle, target)
                    curve.append({"step": step+1, "loss": float(loss.detach()), **scores})
                    print(json.dumps({"kind": kind, "decoder": bundle, "step": step+1, "iou": scores["iou"], "f1": scores["f1"]}), flush=True)
            torch.cuda.synchronize(); d.check_deadline(fit_deadline)
            name = f"{kind}-{bundle}"
            records.append({"name": name, "seconds": time.monotonic()-fit_start,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "curve": curve,
                            "parameters": sum(x.numel() for x in parameters),
                            "overfit_target_met": curve[-1]["iou"] >= .95 and curve[-1]["f1"] >= .97})
            torch.save({"provider": provider.state_dict() if provider is not None else None,
                        "decoder": decoder.state_dict(), "predictions": pred.cpu()}, output/(name+".pt"))
            del provider, decoder, opt, parameters, code, logits, loss, pred
    report = {"registration": env, "status": "learnability_completed", "records": records,
              "elapsed_seconds": time.monotonic()-start+p["preparation_seconds"], "promotion_eligible": False,
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
              "artifacts": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"] = d.base.payload_sha256(report); d.base.write_json(output/"report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze": freeze(args.registration, args.spec_sha, args.source)
    else: run(args.registration, args.source, args.output)


if __name__ == "__main__": main()
