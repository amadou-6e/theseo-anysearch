"""Bounded reference-only screening followed by untouched three-seed assessment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import checkpoint_diagnostic as c

d = c.d
RECIPES = [
    {"name": "baseline", "count": 48, "dynamic": False, "lr": .001, "wd": .01},
    {"name": "dynamic", "count": 48, "dynamic": True, "lr": .001, "wd": .01},
    {"name": "coverage", "count": 192, "dynamic": True, "lr": .001, "wd": .01},
    {"name": "lower_lr", "count": 192, "dynamic": True, "lr": .0003, "wd": .01},
    {"name": "higher_decay", "count": 192, "dynamic": True, "lr": .001, "wd": .1},
    {"name": "higher_lr", "count": 192, "dynamic": True, "lr": .003, "wd": .01},
]
PLAN = {"study_id": "voxel-reference-refinement-v1", "recipes": RECIPES,
        "steps": 2048, "checkpoints": [16,32,64,128,256,512,1024,1536,2048],
        "screen_seed": 0, "confirmation_seeds": [0,1,2], "cap_seconds": 7200,
        "fit_cap_seconds": 600, "prior": [
            "docs/perception-encoder-local-geometry/diagnostics-preregistration.json",
            "docs/perception-encoder-local-geometry/checkpoint-preregistration.json"],
        "base_settings": d.PLAN,
        "ordering_exception": "User authorizes continuing unmerged experiment stack; no integration merges."}


def make_data(phase: str) -> dict:
    plan = {**d.PLAN, "dataset_id": f"voxel-reference-refinement-v1-{phase}-development-1",
            "splits": {"train": 192, "selection": 48, **({"assessment": 48} if phase == "confirmation" else {})},
            "streams": {"train": 36401 if phase == "screen" else 36411,
                        "selection": 36402 if phase == "screen" else 36412, "assessment": 36413}}
    data = d.corpus(plan)
    # Training boundary truth is used only as a loss target for independently redrawn masks.
    data["train"]["boundary"] = torch.tensor(np.stack([
        d.base.compute_geometry_targets(x.numpy().astype(bool), truncation=8.).boundary
        for x in data["train"]["occupancy"]]), dtype=torch.float32)
    return data


def all_data() -> dict:
    data = {phase: make_data(phase) for phase in ("screen", "confirmation")}
    seen = set()
    for path in PLAN["prior"]:
        p = json.loads(Path(path).read_text())["payload"]
        seen.update(h for split in p["data"].values() for h in split["occupancy_hashes"])
    for phase in data.values():
        for rows in phase.values():
            for sha in rows["hashes"]:
                if sha in seen:
                    raise ValueError("duplicate or reused geometry")
                seen.add(sha)
    return data


def data_identity(data: dict) -> dict:
    return {phase: {"banks": d.data_contract(rows), "boundary": d.digest_tensor(rows["train"]["boundary"])}
            for phase, rows in data.items()}


def freeze(path: Path, spec: str, artifacts: Path, source_report: Path):
    start = time.monotonic()
    if len(spec) != 40 or any(x not in "0123456789abcdef" for x in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    contract = d.lg3.artifact_contract(source_report)
    files = {k:v for k,v in contract["files"].items() if k.endswith("encoder.pt")}
    for name, sha in files.items():
        if hashlib.sha256((artifacts/name).read_bytes()).hexdigest() != sha:
            raise ValueError("checkpoint mismatch")
    payload = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"),
               "spec_commit": spec, "spec_path": "projects/theseo-anysearch/python/perception-reference-refinement.md",
               "data": data_identity(all_data()), "files": files, "states": contract["states"],
               "prior_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in PLAN["prior"]},
               "preparation_seconds": time.monotonic()-start}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def dynamic_batch(rows: dict, idx: torch.Tensor, rng: torch.Generator) -> tuple:
    occ = rows["occupancy"][idx]
    hidden = torch.rand(occ.shape, generator=rng, device=occ.device) < .2
    if int(hidden.flatten(1).sum(1).min()) < 256:
        raise ValueError("insufficient dynamic hidden query support")
    priorities = torch.rand(occ.shape, generator=rng, device=occ.device).masked_fill(~hidden, -1)
    queries = priorities.flatten(1).topk(256, dim=1).indices
    inputs = d.visible({"occupancy": occ, "hidden": hidden.unsqueeze(1)})
    targets = rows["boundary"][idx].flatten(1).gather(1, queries)
    return inputs, queries, targets


def train(data: dict, recipe: dict, seed: int, deadline: float):
    torch.manual_seed(36010+seed)
    model = d.Reference().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["wd"])
    rng = torch.Generator(device="cuda").manual_seed(36020+seed)
    masks = torch.Generator(device="cuda").manual_seed(36420+seed)
    rows = data["train"]
    fixed = d.visible(rows)
    sy = data["selection"]["labels"].cpu().numpy()
    best_loss, state, best_step, curve = float("inf"), None, None, []
    for step in range(1,2049):
        d.check_deadline(deadline)
        idx = torch.randint(recipe["count"], (8,), generator=rng, device="cuda")
        if recipe["dynamic"]:
            inputs, queries, targets = dynamic_batch(rows, idx, masks)
        else:
            inputs, queries, targets = fixed[idx], rows["indices"][idx], rows["labels"][idx]
        logits = model(inputs).flatten(1).gather(1, queries)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in PLAN["checkpoints"]:
            value = float(d.logloss_rows(d.ref_predict(model,data["selection"]),sy).mean())
            curve.append({"step":step,"training_loss":float(loss.detach()),"selection_loss":value})
            if value < best_loss:
                best_loss, best_step, state = value, step, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    d.check_deadline(deadline)
    print(json.dumps({"recipe":recipe["name"],"seed":seed,"best_step":best_step,"selection_loss":best_loss}),flush=True)
    return model.eval(), {"recipe":recipe["name"],"seed":seed,"best_step":best_step,"selection_loss":best_loss,"curve":curve}


def select(rows: list) -> str:
    if [r["recipe"] for r in rows] != [r["name"] for r in RECIPES]:
        raise ValueError("complete ordered screen required")
    return min(rows,key=lambda r:r["selection_loss"])["recipe"]


def probe(data: dict, seed: int, artifacts: Path, states: dict, output: Path, deadline: float):
    encoder = d.base.make_encoder(seed,torch.device("cuda"))
    encoder.load_state_dict(torch.load(artifacts/f"joint-{seed}-encoder.pt",map_location="cuda",weights_only=True))
    encoder.eval().requires_grad_(False)
    before = d.base.encoder_state_sha256(encoder)
    if before != states[str(seed)]["final_state_sha256"]:
        raise ValueError("encoder state mismatch")
    features = {}
    # Probe fitting remains on 48 geometries, matching its registered prior budget.
    small = {s:{k:v[:48] if isinstance(v,torch.Tensor) else v for k,v in rows.items()} for s,rows in data.items()}
    with torch.no_grad():
        for split, rows in small.items():
            values=[]
            for start in range(0,len(rows["labels"]),8):
                occ,hidden=rows["occupancy"][start:start+8],rows["hidden"][start:start+8]
                volume=encoder(d.base.VoxelLevel.from_occupancy(occ,unknown_mask=hidden),hidden).local_feature_volume
                values.append(d.base.gather_features(volume,rows["indices"][start:start+8]))
            features[split]=torch.cat(values)
    candidates=[]
    sy,y=(small[s]["labels"].cpu().numpy() for s in ("selection","assessment"))
    for width in [0,32,128]:
        model,mean,scale,curve=d.fit_probe(features,small,seed,width,deadline)
        with torch.no_grad():
            sp=model((features["selection"]-mean)/scale)[...,0].sigmoid().cpu().numpy()
            pred=model((features["assessment"]-mean)/scale)[...,0].sigmoid().cpu().numpy()
        t=d.threshold(sp,sy)
        candidates.append({"width":width,"selection_loss":float(d.logloss_rows(sp,sy).mean()),"assessment":d.evaluate(pred,y,t),"curve":curve})
        torch.save({"state":model.state_dict(),"predictions":pred,"mean":mean,"scale":scale},output/f"probe-{seed}-{width}.pt")
    if d.base.encoder_state_sha256(encoder)!=before:
        raise ValueError("encoder mutated")
    chosen=min(candidates,key=lambda r:r["selection_loss"])
    return {"assessment":chosen["assessment"],"selected_width":chosen["width"],"capacities":candidates,"encoder_state_sha256":before}


def run(registration: Path, artifacts: Path, output: Path):
    started=time.monotonic()
    env=json.loads(registration.read_text()); p=env["payload"]
    if set(env)!={"payload","identity_sha256"} or d.base.payload_sha256(p)!=env["identity_sha256"] or p["plan"]!=PLAN:
        raise ValueError("registration mismatch")
    if d.base.git("diff",p["source_commit"],"--","theseo_anysearch/garden") or d.base.git("ls-files","--others","--exclude-standard","theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    for name,sha in p["files"].items():
        if hashlib.sha256((artifacts/name).read_bytes()).hexdigest()!=sha: raise ValueError("checkpoint mismatch")
    for name,sha in p["prior_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha: raise ValueError("prior identity mismatch")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    deadline=started+7200-p["preparation_seconds"]
    report={"registration":env,"screen":[],"trials":[],"encoder_updates":0}
    try:
        data=all_data()
        if data_identity(data)!=p["data"]: raise ValueError("data mismatch")
        data={phase:{s:{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in rows.items()} for s,rows in splits.items()} for phase,splits in data.items()}
        report["D0"]=d.d0(min(deadline,time.monotonic()+300))
        if report["D0"]["outcome"]!="pass": raise ValueError("D0 failed")
        for recipe in RECIPES:
            model,row=train(data["screen"],recipe,0,min(deadline,time.monotonic()+600))
            report["screen"].append(row)
        chosen=select(report["screen"])
        report["chosen_recipe"]=chosen
        # Persist the selection before any confirmation fitting or assessment.
        d.base.write_json(output/"selection.json",{"chosen":chosen,"screen":report["screen"],"registration":env["identity_sha256"]})
        recipe=next(r for r in RECIPES if r["name"]==chosen)
        confirmation=data["confirmation"]
        sy,y=(confirmation[s]["labels"].cpu().numpy() for s in ("selection","assessment"))
        for seed in [0,1,2]:
            trial={"seed":seed}
            for label,settings in (("baseline",RECIPES[0]),("chosen",recipe)):
                model,row=train(confirmation,settings,seed,min(deadline,time.monotonic()+600))
                sp=d.ref_predict(model,confirmation["selection"]); t=d.threshold(sp,sy)
                pred=d.ref_predict(model,confirmation["assessment"])
                trial[label]={**row,"assessment":d.evaluate(pred,y,t)}
                torch.save({"state":model.state_dict(),"predictions":pred},output/f"{label}-{seed}.pt")
            trial["probe"]=probe(confirmation,seed,artifacts,p["states"],output,min(deadline,time.monotonic()+600))
            report["trials"].append(trial)
        comparison=[{"seed":t["seed"],"best":t["chosen"]["assessment"],"final":t["baseline"]["assessment"],"probe":t["probe"]["assessment"]} for t in report["trials"]]
        report["assessment"]=c.assess(comparison)
        report["validity"]="valid"
    except TimeoutError as exc:
        report.update(validity="valid",assessment={"outcome":"inconclusive","reason":str(exc)})
    except (ValueError,FloatingPointError) as exc:
        report.update(validity="invalid",assessment={"outcome":"repair_required","reason":str(exc)})
    report["elapsed_seconds"]=time.monotonic()-started+p["preparation_seconds"]
    report["environment"]={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__}
    report["artifacts"]={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir() if f.is_file()}
    report["report_payload_sha256"]=d.base.payload_sha256(report)
    d.base.write_json(output/"report.json",report)
    print(json.dumps(report["assessment"]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["freeze","run"])
    parser.add_argument("--registration",type=Path,required=True)
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--source-report",type=Path)
    parser.add_argument("--output",type=Path)
    a=parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if a.command=="freeze": freeze(a.registration,a.spec_sha,a.artifacts,a.source_report)
    else: run(a.registration,a.artifacts,a.output)


if __name__=="__main__": main()
