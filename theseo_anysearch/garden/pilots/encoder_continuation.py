"""Fixed architecture: probe exposure and new joint encoder pretraining."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from . import reference_refinement as r

d=r.d
PLAN={"study_id":"voxel-encoder-continuation-v1", "pretrain_steps":2048,
      "lr":.003,"weight_decay":.01,"seeds":[0,1,2],"cap_seconds":7200,
      "fit_cap_seconds":900,"joint_esdf_weight":10.,"probe_counts":[48,192],
      "base_settings":r.PLAN,
      "prior_registration":"docs/perception-encoder-local-geometry/refinement-preregistration.json"}


def dataset():
    plan={**d.PLAN,"dataset_id":"voxel-encoder-continuation-v1-development-1",
          "splits":{"pretrain":192,"train":192,"selection":48,"assessment":48},
          "streams":{"pretrain":36600,"train":36601,"selection":36602,"assessment":36603}}
    data=d.corpus(plan)
    prior=json.loads(Path(PLAN["prior_registration"]).read_text())["payload"]["data"]
    seen={h for phase in prior.values() for split in phase["banks"].values() for h in split["occupancy_hashes"]}
    for rows in data.values():
        if any(h in seen for h in rows["hashes"]): raise ValueError("prior geometry reused")
    for split in ("pretrain","train"):
        targets=[d.base.compute_geometry_targets(x.numpy().astype(bool),truncation=8.) for x in data[split]["occupancy"]]
        data[split]["boundary"]=torch.tensor(np.stack([x.boundary for x in targets]),dtype=torch.float32)
        data[split]["distance"]=torch.tensor(np.stack([x.signed_distance/8 for x in targets]),dtype=torch.float32)
    return data


def identity(data):
    return {"banks":d.data_contract(data),"targets":{s:{k:d.digest_tensor(data[s][k]) for k in ("boundary","distance")} for s in ("pretrain","train")}}


def freeze(path,spec,artifacts,source_report):
    start=time.monotonic()
    if len(spec)!=40 or any(x not in "0123456789abcdef" for x in spec): raise ValueError("full spec SHA required")
    if d.base.git("status","--porcelain","--","theseo_anysearch/garden"): raise ValueError("commit source first")
    contract=d.lg3.artifact_contract(source_report)
    files={k:v for k,v in contract["files"].items() if k.endswith("encoder.pt")}
    for name,sha in files.items():
        if hashlib.sha256((artifacts/name).read_bytes()).hexdigest()!=sha: raise ValueError("checkpoint mismatch")
    p={"plan":PLAN,"source_commit":d.base.git("rev-parse","HEAD"),"spec_commit":spec,
       "spec_path":"projects/theseo-anysearch/python/perception-encoder-continuation.md",
       "data":identity(dataset()),"files":files,"states":contract["states"],
       "prior_sha256":hashlib.sha256(Path(PLAN["prior_registration"]).read_bytes()).hexdigest(),
       "preparation_seconds":time.monotonic()-start}
    d.base.write_json(path,{"payload":p,"identity_sha256":d.base.payload_sha256(p)})


def pretrain(rows,seed,deadline):
    model=d.base.make_encoder(seed,torch.device("cuda")).train()
    head=nn.Conv3d(8,2,1).cuda()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(head.parameters()),lr=.003,weight_decay=.01)
    rng=torch.Generator(device="cuda").manual_seed(36610+seed)
    curve=[]
    for step in range(2048):
        d.check_deadline(deadline)
        idx=torch.randint(192,(8,),generator=rng,device="cuda")
        occ=rows["occupancy"][idx]
        hidden=(torch.rand(occ.shape,generator=rng,device="cuda")<.2).unsqueeze(1)
        out=model(d.base.VoxelLevel.from_occupancy(occ,unknown_mask=hidden),hidden).local_feature_volume
        loss,ol,dl=d.lg3.prior.objective_loss(head(out),occ,rows["distance"][idx],hidden,"joint")
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite encoder loss")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step+1)%256==0:
            curve.append({"step":step+1,"loss":float(loss.detach()),"occupancy_loss":float(ol.detach()),"esdf_loss":float(dl.detach())})
    return model.eval().requires_grad_(False),curve


def assess(trials):
    if sorted(t["seed"] for t in trials)!=[0,1,2]: return {"outcome":"inconclusive"}
    get=lambda name:[t[name]["assessment"] for t in trials]
    exposure=r.c.paired_gain(get("old192"),get("old48"))
    retraining=r.c.paired_gain(get("new192"),get("old192"))
    reference=r.c.paired_gain(get("new192"),get("reference"))
    direction=("validate_distance_retention_and_new_families" if retraining["ci95"][0]>0 else
               "retain_encoder_and_investigate_objective_readout_gap")
    return {"exposure_loss_gain":exposure,"retraining_loss_gain":retraining,
            "new_encoder_vs_reference_loss_gain":reference,"next_action":direction,"promotion_eligible":False}


def run(registration,artifacts,output):
    started=time.monotonic(); env=json.loads(registration.read_text()); p=env["payload"]
    if set(env)!={"payload","identity_sha256"} or d.base.payload_sha256(p)!=env["identity_sha256"] or p["plan"]!=PLAN: raise ValueError("registration mismatch")
    if d.base.git("diff",p["source_commit"],"--","theseo_anysearch/garden") or d.base.git("ls-files","--others","--exclude-standard","theseo_anysearch/garden"): raise ValueError("source mismatch")
    if hashlib.sha256(Path(PLAN["prior_registration"]).read_bytes()).hexdigest()!=p["prior_sha256"]: raise ValueError("prior mismatch")
    for name,sha in p["files"].items():
        if hashlib.sha256((artifacts/name).read_bytes()).hexdigest()!=sha: raise ValueError("checkpoint mismatch")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    deadline=started+7200-p["preparation_seconds"]
    report={"registration":env,"trials":[]}
    try:
        data=dataset()
        if identity(data)!=p["data"]: raise ValueError("data mismatch")
        data={s:{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in rows.items()} for s,rows in data.items()}
        report["D0"]=d.d0(min(deadline,time.monotonic()+300))
        if report["D0"]["outcome"]!="pass": raise ValueError("D0 failed")
        fitdata={k:v for k,v in data.items() if k!="pretrain"}
        sy,y=(data[s]["labels"].cpu().numpy() for s in ("selection","assessment"))
        for seed in [0,1,2]:
            trial={"seed":seed}
            reference,row=r.train(fitdata,r.RECIPES[-1],seed,min(deadline,time.monotonic()+900))
            sp=d.ref_predict(reference,data["selection"]); pred=d.ref_predict(reference,data["assessment"])
            trial["reference"]={**row,"assessment":d.evaluate(pred,y,d.threshold(sp,sy))}
            torch.save({"state":reference.state_dict(),"predictions":pred},output/f"reference-{seed}.pt")
            for count in [48,192]:
                folder=output/f"old{count}-{seed}"; folder.mkdir()
                trial[f"old{count}"]=r.probe(fitdata,seed,artifacts,p["states"],folder,min(deadline,time.monotonic()+900),training_count=count)
            model,curve=pretrain(data["pretrain"],seed,min(deadline,time.monotonic()+900))
            new=output/f"new-{seed}"; new.mkdir()
            torch.save(model.state_dict(),new/f"joint-{seed}-encoder.pt")
            state=d.base.encoder_state_sha256(model)
            trial["new192"]=r.probe(fitdata,seed,new,{str(seed):{"final_state_sha256":state}},new,min(deadline,time.monotonic()+900),training_count=192)
            trial["pretraining"]={"updates":2048,"curve":curve,"encoder_state_sha256":state}
            report["trials"].append(trial)
            print(json.dumps({"seed":seed,**{k:trial[k]["assessment"]["f1_calibrated"] for k in ("old48","old192","new192","reference")}}),flush=True)
        report["assessment"]=assess(report["trials"]); report["validity"]="valid"
    except TimeoutError as exc: report.update(validity="valid",assessment={"outcome":"inconclusive","reason":str(exc)})
    except (ValueError,FloatingPointError) as exc: report.update(validity="invalid",assessment={"outcome":"repair_required","reason":str(exc)})
    report["elapsed_seconds"]=time.monotonic()-started+p["preparation_seconds"]
    report["environment"]={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__}
    report["artifacts"]={str(f.relative_to(output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in output.rglob("*.pt")}
    report["report_payload_sha256"]=d.base.payload_sha256(report)
    d.base.write_json(output/"report.json",report); print(json.dumps(report["assessment"]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=["freeze","run"])
    p.add_argument("--registration",type=Path,required=True); p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--spec-sha"); p.add_argument("--source-report",type=Path); p.add_argument("--output",type=Path)
    a=p.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if a.command=="freeze": freeze(a.registration,a.spec_sha,a.artifacts,a.source_report)
    else: run(a.registration,a.artifacts,a.output)

if __name__=="__main__": main()
