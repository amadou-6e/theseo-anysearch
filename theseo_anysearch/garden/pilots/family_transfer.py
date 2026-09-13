"""Frozen old/new encoder comparison on two held-out generator families."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import distance_retention as prior

d=prior.d
FAMILIES=("shells","trig_labyrinth")
PLAN={"study_id":"voxel-family-transfer-v1","seeds":[0,1,2],"probe_steps":1024,
      "counts":{"train":192,"selection":48,"assessment":48},
      "families":list(FAMILIES),"hours_cap":1,"classification_margin":.03,"distance_margin":.02,
      "source_report":prior.PLAN["source_report"],"prior_registration":"docs/perception-encoder-local-geometry/retention-preregistration.json"}


def generate(gid,family,fraction):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(gid.encode()).digest()[:8],"little"))
    xyz=np.stack(np.meshgrid(*([np.linspace(-1,1,17)]*3),indexing="ij"))
    if family=="shells":
        center=rng.uniform(-.3,.3,3)[:,None,None,None]
        field=np.abs(np.linalg.norm(xyz-center,axis=0)-rng.uniform(.45,.85))
    elif family=="trig_labyrinth":
        x,y,z=xyz*rng.uniform(2.5,5.5,3)[:,None,None,None]+rng.uniform(-np.pi,np.pi,3)[:,None,None,None]
        field=np.sin(x)*np.cos(y)+np.sin(y)*np.cos(z)+np.sin(z)*np.cos(x)
    else: raise ValueError("unknown transfer family")
    return field<=np.quantile(field,fraction)


def data():
    rows={}; seen=set()
    previous=json.loads(Path(PLAN["prior_registration"]).read_text())["payload"]["identity"]["data"]
    seen.update(h for s in previous.values() for h in s["occupancy_hashes"])
    for split,count in PLAN["counts"].items():
        values={k:[] for k in ("occupancy","boundary","distance","ids","hashes","families")}
        for i in range(count):
            gid=f"voxel-family-transfer-v1-development-1-{split}-{i:03d}"
            family=FAMILIES[(i%6)//3] if split=="assessment" else d.lg3.FAMILIES[(i%12)//3]
            grid=generate(gid,family,[.08,.16,.28][i%3]) if split=="assessment" else d.lg3.generate(gid,family,[.08,.16,.28][i%3])
            sha=hashlib.sha256(grid.tobytes()).hexdigest()
            if sha in seen: raise ValueError("duplicate geometry")
            seen.add(sha); target=d.base.compute_geometry_targets(grid,truncation=8.)
            for k,v in (("occupancy",grid),("boundary",target.boundary),("distance",target.signed_distance/8),("ids",gid),("hashes",sha),("families",family)): values[k].append(v)
        for k in ("occupancy","boundary","distance"): values[k]=torch.tensor(np.stack(values[k]),dtype=torch.float32)
        rows[split]=values
    banks={s:d.base.query_bank(v,"probe" if s=="train" else "evaluation") for s,v in rows.items()}
    for split in rows:
        for family in set(rows[split]["families"]):
            idx=[i for i,f in enumerate(rows[split]["families"]) if f==family]
            for task in d.base.TASKS[:2]:
                y=banks[split]["targets"][task][idx]
                if min(float(y.sum()),float((1-y).sum()))<20: raise ValueError("insufficient family class support")
    return rows,banks


def identity(rows,banks):
    return {s:{"ids":v["ids"],"hashes":v["hashes"],"families":v["families"],"queries":banks[s]["sha256"]} for s,v in rows.items()}


def freeze(path,spec,oldroot,newroot):
    if len(spec)!=40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if d.base.git("status","--porcelain","--","theseo_anysearch/garden"): raise ValueError("commit source first")
    p={"plan":PLAN,"source_commit":d.base.git("rev-parse","HEAD"),"spec_commit":spec,
       "spec_path":"projects/theseo-anysearch/python/perception-family-transfer.md",
       "encoders":prior.contracts(oldroot,newroot),"identity":identity(*data()),
       "prior_hash":hashlib.sha256(Path(PLAN["prior_registration"]).read_bytes()).hexdigest(),
       "report_hash":hashlib.sha256(Path(PLAN["source_report"]).read_bytes()).hexdigest()}
    d.base.write_json(path,{"payload":p,"identity_sha256":d.base.payload_sha256(p)})


def assess(trials):
    results={}
    for task in d.base.TASKS:
        for f,family in enumerate(FAMILIES):
            selected={arm:sorted([t for t in trials if t["arm"]==arm and t["task"]==task],key=lambda x:x["seed"]) for arm in ("old","new")}
            if any([t["seed"] for t in v]!=[0,1,2] for v in selected.values()): return {"outcome":"inconclusive"}
            idx=np.array([i for i in range(48) if (i%6)//3==f])
            stats={arm:[np.asarray(t["statistics"])[idx] for t in v] for arm,v in selected.items()}
            scores={arm:[d.base.score(v,task) for v in vlist] for arm,vlist in stats.items()}
            higher=task in d.base.TASKS[:2]
            rng=np.random.default_rng(370)
            strata=[np.arange(k,24,3) for k in range(3)]
            gains=[]
            for _ in range(2000):
                draw=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
                gains.append(float(np.mean([(d.base.score(n[draw],task)-d.base.score(o[draw],task))*(1 if higher else -1) for n,o in zip(stats["new"],stats["old"])])))
            ci=np.quantile(gains,[.025,.975]).tolist()
            bar=d.base.PLAN["absolute_bars"][task]
            absolute=all(v>=bar if higher else v<=bar for v in scores["new"])
            results[f"{family}/{task}"]={"scores":scores,"improvement_ci95":ci,"absolute_pass":absolute,
                "pass":absolute and ci[0]>=-(.03 if higher else .02)}
    return {"outcome":"transfer_supported" if all(r["pass"] for r in results.values()) else "transfer_unconfirmed","components":results,"promotion_eligible":False}


def run(path,oldroot,newroot,output):
    start=time.monotonic(); env=json.loads(path.read_text()); p=env["payload"]
    if set(env)!={"payload","identity_sha256"} or d.base.payload_sha256(p)!=env["identity_sha256"] or p["plan"]!=PLAN: raise ValueError("registration mismatch")
    if d.base.git("diff",p["source_commit"],"--","theseo_anysearch/garden") or d.base.git("ls-files","--others","--exclude-standard","theseo_anysearch/garden"): raise ValueError("source mismatch")
    if hashlib.sha256(Path(PLAN["prior_registration"]).read_bytes()).hexdigest()!=p["prior_hash"] or hashlib.sha256(Path(PLAN["source_report"]).read_bytes()).hexdigest()!=p["report_hash"] or prior.contracts(oldroot,newroot)!=p["encoders"]: raise ValueError("input artifacts mismatch")
    rows,banks=data()
    if identity(rows,banks)!=p["identity"]: raise ValueError("data mismatch")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    rows={s:{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in x.items()} for s,x in rows.items()}
    for b in banks.values():
        b["hidden"]=b["hidden"].cuda()
        for k in ("targets","indices"): b[k]={t:v.cuda() for t,v in b[k].items()}
    trials=[]
    for entry in p["encoders"]:
        d.check_deadline(start+3600); seed,arm=entry["seed"],entry["arm"]
        source=oldroot/f"joint-{seed}-encoder.pt" if arm=="old" else newroot/f"new-{seed}"/f"joint-{seed}-encoder.pt"
        model=d.base.make_encoder(seed,torch.device("cuda")); model.load_state_dict(torch.load(source,map_location="cuda",weights_only=True)); model.eval().requires_grad_(False)
        if d.base.encoder_state_sha256(model)!=entry["state_sha256"]: raise ValueError("state mismatch")
        features={s:d.base.extract(model,rows[s],banks[s]) for s in rows}
        for task in d.base.TASKS:
            fitted=d.base.fit_probe(features["train"][task],banks["train"]["targets"][task],task,seed,steps=1024)
            pred=d.base.predict(fitted,features["assessment"][task],task); threshold=.5
            if task in d.base.TASKS[:2]:
                sp=d.base.predict(fitted,features["selection"][task],task).cpu().numpy()
                threshold=d.threshold(sp,banks["selection"]["targets"][task].cpu().numpy())
                scored=(pred>=threshold).float()
            else: scored=pred
            stats=d.base.sufficient_statistics(scored,banks["assessment"]["targets"][task],task)
            trials.append({"seed":seed,"arm":arm,"task":task,"threshold":threshold,"statistics":stats})
            torch.save({"state":fitted[0].state_dict(),"predictions":pred.cpu()},output/f"{arm}-{seed}-{task}.pt")
            d.check_deadline(start+3600)
        if d.base.encoder_state_sha256(model)!=entry["state_sha256"]: raise ValueError("encoder mutated")
        print(json.dumps({"completed":arm,"seed":seed}),flush=True)
    report={"registration":env,"trials":trials,"assessment":assess(trials),"encoder_updates":0,"elapsed_seconds":time.monotonic()-start,
        "environment":{"gpu":torch.cuda.get_device_name(),"torch":torch.__version__},"artifacts":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"]=d.base.payload_sha256(report); d.base.write_json(output/"report.json",report); print(json.dumps(report["assessment"]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=["freeze","run"])
    p.add_argument("--registration",type=Path,required=True); p.add_argument("--old",type=Path,required=True); p.add_argument("--new",type=Path,required=True)
    p.add_argument("--spec-sha"); p.add_argument("--output",type=Path)
    a=p.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if a.command=="freeze": freeze(a.registration,a.spec_sha,a.old,a.new)
    else: run(a.registration,a.old,a.new,a.output)

if __name__=="__main__": main()
