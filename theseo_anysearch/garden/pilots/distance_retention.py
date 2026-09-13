"""Fresh distance retention comparison of old and newly frozen encoders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import encoder_continuation as e

d=e.d
PLAN={"study_id":"voxel-distance-retention-v1","margin":.02,"seeds":[0,1,2],
      "probe_steps":1024,"hours_cap":1,"tasks":["clearance_nmae","recovery_nmae"],
      "source_report":"docs/perception-encoder-local-geometry/continuation-report.json"}


def data():
    plan={**d.PLAN,"dataset_id":"voxel-distance-retention-v1-development-1",
          "splits":{"train":192,"assessment":48},"streams":{"train":36801,"assessment":36802}}
    rows=d.corpus(plan)
    prior=json.loads(Path(PLAN["source_report"]).read_text())["registration"]["payload"]["data"]["banks"]
    old={h for split in prior.values() for h in split["occupancy_hashes"]}
    for split in rows.values():
        if any(h in old for h in split["hashes"]): raise ValueError("old geometry reused")
        targets=[d.base.compute_geometry_targets(x.numpy().astype(bool),truncation=8.) for x in split["occupancy"]]
        split["boundary"]=torch.tensor(np.stack([x.boundary for x in targets]),dtype=torch.float32)
        split["distance"]=torch.tensor(np.stack([x.signed_distance/8 for x in targets]),dtype=torch.float32)
    # Existing query algorithm, CPU generated: observed-free clearance, hidden-free recovery.
    banks={s:d.base.query_bank(v,"probe" if s=="train" else "evaluation") for s,v in rows.items()}
    return rows,banks


def identity(rows,banks):
    return {"data":d.data_contract(rows),"query_hashes":{s:b["sha256"] for s,b in banks.items()},
            "distances":{s:d.digest_tensor(v["distance"]) for s,v in rows.items()}}


def contracts(oldroot,newroot):
    report=json.loads(Path(PLAN["source_report"]).read_text()); sha=report.pop("report_payload_sha256")
    if d.base.payload_sha256(report)!=sha: raise ValueError("source report corrupt")
    hashes={k.replace('\\','/'):v for k,v in report["artifacts"].items()}
    records=[]
    for seed in [0,1,2]:
        for arm in ("old","new"):
            name=f"joint-{seed}-encoder.pt"
            path=oldroot/name if arm=="old" else newroot/f"new-{seed}"/name
            expected=report["registration"]["payload"]["files"][name] if arm=="old" else hashes[f"new-{seed}/{name}"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError("encoder file mismatch")
            state=report["registration"]["payload"]["states"][str(seed)]["final_state_sha256"] if arm=="old" else next(t for t in report["trials"] if t["seed"]==seed)["pretraining"]["encoder_state_sha256"]
            records.append({"seed":seed,"arm":arm,"file_sha256":expected,"state_sha256":state})
    return records


def freeze(path,spec,oldroot,newroot):
    if len(spec)!=40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if d.base.git("status","--porcelain","--","theseo_anysearch/garden"): raise ValueError("commit source first")
    p={"plan":PLAN,"source_commit":d.base.git("rev-parse","HEAD"),"spec_commit":spec,
       "spec_path":"projects/theseo-anysearch/python/perception-distance-retention.md",
       "encoders":contracts(oldroot,newroot),"identity":identity(*data()),
       "source_report_sha256":hashlib.sha256(Path(PLAN["source_report"]).read_bytes()).hexdigest()}
    d.base.write_json(path,{"payload":p,"identity_sha256":d.base.payload_sha256(p)})


def assess(trials):
    results={}
    for task in PLAN["tasks"]:
        old=[t for t in trials if t["arm"]=="old" and t["task"]==task]
        new=[t for t in trials if t["arm"]=="new" and t["task"]==task]
        old.sort(key=lambda t:t["seed"]); new.sort(key=lambda t:t["seed"])
        if [t["seed"] for t in old]!=[0,1,2] or [t["seed"] for t in new]!=[0,1,2]: return {"outcome":"inconclusive"}
        delta=np.mean([np.array(n["geometry_nmae"])-np.array(o["geometry_nmae"]) for o,n in zip(old,new)],axis=0)
        rng=np.random.default_rng(368)
        strata=[np.arange(i,48,12) for i in range(12)]
        draws=[float(delta[np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])].mean()) for _ in range(2000)]
        interval=np.quantile(draws,[.025,.975]).tolist()
        scores=[t["nmae"] for t in new]; differences=[n["nmae"]-o["nmae"] for o,n in zip(old,new)]
        passed=interval[1]<=.02 and max(differences)<=.02 and max(scores)<=(.15 if task=="clearance_nmae" else .20)
        results[task]={"old_scores":[t["nmae"] for t in old],"new_scores":scores,"new_minus_old_ci95":interval,"retained":passed}
    return {"outcome":"distance_retained" if all(x["retained"] for x in results.values()) else "distance_retention_unconfirmed","tasks":results,"promotion_eligible":False}


def run(path,oldroot,newroot,output):
    start=time.monotonic(); env=json.loads(path.read_text()); p=env["payload"]
    if d.base.payload_sha256(p)!=env["identity_sha256"] or p["plan"]!=PLAN: raise ValueError("registration mismatch")
    if d.base.git("diff",p["source_commit"],"--","theseo_anysearch/garden") or d.base.git("ls-files","--others","--exclude-standard","theseo_anysearch/garden"): raise ValueError("source mismatch")
    if hashlib.sha256(Path(PLAN["source_report"]).read_bytes()).hexdigest()!=p["source_report_sha256"] or contracts(oldroot,newroot)!=p["encoders"]: raise ValueError("source artifacts mismatch")
    rows,banks=data()
    if identity(rows,banks)!=p["identity"]: raise ValueError("data mismatch")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    rows={s:{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in x.items()} for s,x in rows.items()}
    for bank in banks.values():
        bank["hidden"]=bank["hidden"].cuda()
        for k in ("indices","targets"): bank[k]={t:v.cuda() for t,v in bank[k].items()}
    trials=[]
    for entry in p["encoders"]:
        d.check_deadline(start+3600)
        seed,arm=entry["seed"],entry["arm"]
        source=oldroot/f"joint-{seed}-encoder.pt" if arm=="old" else newroot/f"new-{seed}"/f"joint-{seed}-encoder.pt"
        model=d.base.make_encoder(seed,torch.device("cuda")); model.load_state_dict(torch.load(source,weights_only=True,map_location="cuda")); model.eval().requires_grad_(False)
        if d.base.encoder_state_sha256(model)!=entry["state_sha256"]: raise ValueError("state mismatch")
        features={s:d.base.extract(model,rows[s],banks[s]) for s in rows}
        for task in PLAN["tasks"]:
            fitted=d.base.fit_probe(features["train"][task],banks["train"]["targets"][task],task,seed,steps=1024)
            pred=d.base.predict(fitted,features["assessment"][task],task)
            errors=(pred-banks["assessment"]["targets"][task]).abs().mean(1).cpu().numpy()
            trials.append({"seed":seed,"arm":arm,"task":task,"nmae":float(errors.mean()),"geometry_nmae":errors.tolist()})
            torch.save({"predictions":pred.cpu(),"state":fitted[0].state_dict()},output/f"{arm}-{seed}-{task}.pt")
            d.check_deadline(start+3600)
        if d.base.encoder_state_sha256(model)!=entry["state_sha256"]: raise ValueError("frozen encoder mutated")
    report={"registration":env,"trials":trials,"assessment":assess(trials),"encoder_updates":0,"elapsed_seconds":time.monotonic()-start,
            "artifacts":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"]=d.base.payload_sha256(report); d.base.write_json(output/"report.json",report)
    print(json.dumps(report["assessment"]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=["freeze","run"])
    p.add_argument("--registration",type=Path,required=True); p.add_argument("--old",type=Path,required=True); p.add_argument("--new",type=Path,required=True)
    p.add_argument("--spec-sha"); p.add_argument("--output",type=Path)
    a=p.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if a.command=="freeze": freeze(a.registration,a.spec_sha,a.old,a.new)
    else: run(a.registration,a.old,a.new,a.output)

if __name__=="__main__": main()
