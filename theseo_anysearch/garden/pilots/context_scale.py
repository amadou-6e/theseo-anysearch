"""Paired 17/33 crops at fixed voxel spacing, shared queries and parent targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from . import family_transfer as previous

d=previous.d
PLAN={"study_id":"voxel-context-scale-v1","sides":[17,33],"parent_side":49,
      "counts":{"pretrain":48,"train":48,"selection":24,"assessment":48},
      "seeds":[0,1,2],"encoder_steps":512,"encoder_batch":4,"lr":.003,"weight_decay":.01,
      "probe_steps":1024,"queries":256,"cap_seconds":7200,
      "families":["random_field","oblique_sheets"],"bootstrap_seed":372,
      "source_report":"docs/perception-encoder-local-geometry/continuation-report.json"}


def crop(x,side):
    offset=(x.shape[-1]-side)//2
    return x[...,offset:offset+side,offset:offset+side,offset:offset+side]


def scene(gid,family,fraction):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(gid.encode()).digest()[:8],"little"))
    if family=="random_field":
        field=gaussian_filter(rng.normal(size=(49,49,49)),sigma=1.5,mode="reflect")
    elif family=="oblique_sheets":
        xyz=np.stack(np.meshgrid(*([np.arange(-24,25)/8]*3),indexing="ij"))
        direction=rng.normal(size=3); direction/=np.linalg.norm(direction)
        field=np.abs(np.sin(rng.uniform(4,8)*(xyz*direction[:,None,None,None]).sum(0)+rng.uniform(-np.pi,np.pi)))
    else: raise ValueError("unknown scale family")
    return field<=np.quantile(field,fraction)


def data():
    result={}; seen=set()
    for split,count in PLAN["counts"].items():
        arrays={k:[] for k in ("occupancy","boundary","distance","hidden")}
        ids=[]; hashes=[]
        for i in range(count):
            gid=f"voxel-context-scale-v1-data-{split}-{i:03d}"
            occ=scene(gid,PLAN["families"][(i%6)//3],[.08,.16,.28][i%3])
            sha=hashlib.sha256(occ.tobytes()).hexdigest()
            if sha in seen: raise ValueError("duplicate scene")
            seen.add(sha); ids.append(gid); hashes.append(sha)
            t=d.base.compute_geometry_targets(occ,truncation=8.)
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256((gid+"-mask").encode()).digest()[:8],"little"))
            for k,v in (("occupancy",occ),("boundary",t.boundary),("distance",t.signed_distance/8),("hidden",rng.random(occ.shape)<.2)):
                arrays[k].append(crop(v,33))
        tensors={k:torch.tensor(np.stack(v),dtype=torch.bool if k=="hidden" else torch.float32) for k,v in arrays.items()}
        hidden=crop(tensors["hidden"],17); free=crop(tensors["occupancy"],17)<.5
        masks={"occupied_iou":hidden,"boundary_f1":hidden,"clearance_nmae":~hidden&free,"recovery_nmae":hidden&free}
        indices={}; targets={}; rng=torch.Generator().manual_seed(37200+list(PLAN["counts"]).index(split))
        for task in d.base.TASKS:
            choices=[]
            for mask in masks[task]:
                cells=torch.where(mask.flatten())[0]
                if not len(cells): raise ValueError("empty central query pool")
                draw=torch.randperm(len(cells),generator=rng)[:256] if len(cells)>=256 else torch.randint(len(cells),(256,),generator=rng)
                choices.append(cells[draw])
            indices[task]=torch.stack(choices)
            truth=crop(tensors["occupancy"] if task=="occupied_iou" else tensors["boundary"] if task=="boundary_f1" else tensors["distance"].clamp_min(0),17)
            targets[task]=truth.flatten(1).gather(1,indices[task])
            if task in d.base.TASKS[:2]:
                for f in range(2):
                    y=targets[task][[i for i in range(count) if (i%6)//3==f]]
                    if min(float(y.sum()),float((1-y).sum()))<20: raise ValueError("insufficient family labels")
        result[split]={**tensors,"ids":ids,"hashes":hashes,"indices":indices,"targets":targets}
    return result


def identity(data):
    return {s:{"ids":v["ids"],"hashes":v["hashes"],"arrays":{k:d.digest_tensor(v[k]) for k in ("occupancy","boundary","distance","hidden")},
               "indices":{k:d.digest_tensor(x) for k,x in v["indices"].items()},"targets":{k:d.digest_tensor(x) for k,x in v["targets"].items()}} for s,v in data.items()}


def sources(root):
    r=json.loads(Path(PLAN["source_report"]).read_text()); sha=r.pop("report_payload_sha256")
    if d.base.payload_sha256(r)!=sha: raise ValueError("source report mismatch")
    hashes={k.replace('\\','/'):v for k,v in r["artifacts"].items()}
    records=[]
    for seed in [0,1,2]:
        name=f"new-{seed}/joint-{seed}-encoder.pt"
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=hashes[name]: raise ValueError("checkpoint mismatch")
        records.append({"seed":seed,"path":name,"file_hash":hashes[name],"state_hash":next(t for t in r["trials"] if t["seed"]==seed)["pretraining"]["encoder_state_sha256"]})
    return records


def freeze(path,spec,root):
    start=time.monotonic()
    if len(spec)!=40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if d.base.git("status","--porcelain","--","theseo_anysearch/garden"): raise ValueError("commit source first")
    p={"plan":PLAN,"source_commit":d.base.git("rev-parse","HEAD"),"spec_commit":spec,
       "spec_path":"projects/theseo-anysearch/python/perception-context-scale.md","encoders":sources(root),
       "identity":identity(data()),"preparation_seconds":time.monotonic()-start}
    d.base.write_json(path,{"payload":p,"identity_sha256":d.base.payload_sha256(p)})


def mapped_indices(indices,side):
    offset=(side-17)//2
    return (indices//289+offset)*side*side+((indices//17)%17+offset)*side+(indices%17+offset)


def train(rows,side,seed,deadline):
    model=d.base.make_encoder(seed,torch.device("cuda")).train(); head=nn.Conv3d(8,2,1).cuda()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(head.parameters()),lr=.003,weight_decay=.01)
    rng=torch.Generator(device="cuda").manual_seed(37210+seed); curve=[]
    torch.cuda.synchronize(); start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for step in range(512):
        d.check_deadline(deadline)
        idx=torch.randint(48,(4,),generator=rng,device="cuda")
        occ=crop(rows["occupancy"][idx],side)
        mask=crop(torch.rand((4,33,33,33),generator=rng,device="cuda")<.2,side).unsqueeze(1)
        out=model(d.base.VoxelLevel.from_occupancy(occ,unknown_mask=mask),mask).local_feature_volume
        loss,_,_=d.lg3.prior.objective_loss(head(out),occ,crop(rows["distance"][idx],side),mask,"joint")
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step+1)%128==0: curve.append({"step":step+1,"loss":float(loss.detach())})
    torch.cuda.synchronize()
    return model.eval().requires_grad_(False),{"updates":512,"seconds":time.monotonic()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"curve":curve}


@torch.no_grad()
def features(model,rows,side):
    result={task:[] for task in d.base.TASKS}
    for start in range(0,len(rows["occupancy"]),4):
        occ=crop(rows["occupancy"][start:start+4],side); mask=crop(rows["hidden"][start:start+4],side).unsqueeze(1)
        volume=model(d.base.VoxelLevel.from_occupancy(occ,unknown_mask=mask),mask).local_feature_volume
        for task in result: result[task].append(d.base.gather_features(volume,mapped_indices(rows["indices"][task][start:start+4],side)))
    return {k:torch.cat(v) for k,v in result.items()}


@torch.no_grad()
def profile(model,rows,side):
    occ=crop(rows["occupancy"][:4],side); mask=crop(rows["hidden"][:4],side).unsqueeze(1)
    level=d.base.VoxelLevel.from_occupancy(occ,unknown_mask=mask)
    for _ in range(3): model(level,mask)
    torch.cuda.synchronize(); baseline=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
    for _ in range(20): model(level,mask)
    torch.cuda.synchronize(); seconds=time.monotonic()-start
    return {"batch":4,"iterations":20,"crops_per_second":80/seconds,"ms_per_batch":seconds*50,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"incremental_peak_bytes":torch.cuda.max_memory_allocated()-baseline}


def compare(trials,a,b):
    results={}
    for task in d.base.TASKS:
        aa=sorted([t for t in trials if t["arm"]==a and t["task"]==task],key=lambda t:t["seed"])
        bb=sorted([t for t in trials if t["arm"]==b and t["task"]==task],key=lambda t:t["seed"])
        if [t["seed"] for t in aa]!=[0,1,2] or [t["seed"] for t in bb]!=[0,1,2]: raise ValueError("incomplete comparison")
        av=[np.asarray(t["statistics"]) for t in aa]; bv=[np.asarray(t["statistics"]) for t in bb]
        rng=np.random.default_rng(372); strata=[np.arange(i,48,6) for i in range(6)]; higher=task in d.base.TASKS[:2]
        gains=[]
        for _ in range(2000):
            draw=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
            gains.append(float(np.mean([(d.base.score(x[draw],task)-d.base.score(y[draw],task))*(1 if higher else -1) for x,y in zip(av,bv)])))
        ci=np.quantile(gains,[.025,.975]).tolist()
        results[task]={"a_scores":[d.base.score(x,task) for x in av],"b_scores":[d.base.score(x,task) for x in bv],"gain_ci95":ci,"noninferior":ci[0]>=-(.03 if higher else .02)}
    return results


def run(path,sourceroot,output):
    started=time.monotonic(); env=json.loads(path.read_text()); p=env["payload"]
    if set(env)!={"payload","identity_sha256"} or d.base.payload_sha256(p)!=env["identity_sha256"] or p["plan"]!=PLAN or sources(sourceroot)!=p["encoders"]: raise ValueError("registration mismatch")
    if d.base.git("diff",p["source_commit"],"--","theseo_anysearch/garden") or d.base.git("ls-files","--others","--exclude-standard","theseo_anysearch/garden"): raise ValueError("source mismatch")
    rows=data()
    if identity(rows)!=p["identity"]: raise ValueError("data mismatch")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for v in rows.values():
        for k in ("occupancy","boundary","distance","hidden"): v[k]=v[k].cuda()
        for k in ("indices","targets"): v[k]={t:x.cuda() for t,x in v[k].items()}
    deadline=started+7200-p["preparation_seconds"]; trials=[]; resources=[]
    for seed in [0,1,2]:
        for mode in ("frozen","retrained"):
            for side in (17,33):
                d.check_deadline(deadline); arm=f"{mode}{side}"
                if mode=="frozen":
                    model=d.base.make_encoder(seed,torch.device("cuda")); model.load_state_dict(torch.load(sourceroot/p["encoders"][seed]["path"],map_location="cuda",weights_only=True)); model.eval().requires_grad_(False); training=None
                    if d.base.encoder_state_sha256(model)!=p["encoders"][seed]["state_hash"]: raise ValueError("state mismatch")
                else: model,training=train(rows["pretrain"],side,seed,min(deadline,time.monotonic()+1200))
                state=d.base.encoder_state_sha256(model); resource={"seed":seed,"arm":arm,"inference":profile(model,rows["assessment"],side),"training":training}
                feats={s:features(model,rows[s],side) for s in ("train","selection","assessment")}
                for task in d.base.TASKS:
                    fitted=d.base.fit_probe(feats["train"][task],rows["train"]["targets"][task],task,seed,steps=1024)
                    pred=d.base.predict(fitted,feats["assessment"][task],task); threshold=.5
                    if task in d.base.TASKS[:2]:
                        sp=d.base.predict(fitted,feats["selection"][task],task).cpu().numpy(); threshold=d.threshold(sp,rows["selection"]["targets"][task].cpu().numpy()); scored=(pred>=threshold).float()
                    else: scored=pred
                    stats=d.base.sufficient_statistics(scored,rows["assessment"]["targets"][task],task)
                    families={family:d.base.score(np.asarray(stats)[[i for i in range(48) if (i%6)//3==f]],task) for f,family in enumerate(PLAN["families"])}
                    trials.append({"seed":seed,"arm":arm,"task":task,"threshold":threshold,"statistics":stats,"families":families})
                    torch.save({"predictions":pred.cpu(),"state":fitted[0].state_dict()},output/f"{arm}-{seed}-{task}.pt"); d.check_deadline(deadline)
                if d.base.encoder_state_sha256(model)!=state: raise ValueError("frozen evaluation mutated model")
                resource["state_hash"]=state; resources.append(resource)
                if mode=="retrained": torch.save(model.state_dict(),output/f"{arm}-{seed}-encoder.pt")
                print(json.dumps({"completed":arm,"seed":seed}),flush=True)
                del model,feats
    comparisons={a+"_vs_"+b:compare(trials,a,b) for a,b in (("frozen33","frozen17"),("retrained33","retrained17"),("retrained33","frozen33"))}
    report={"registration":env,"trials":trials,"resources":resources,"comparisons":comparisons,"elapsed_seconds":time.monotonic()-started+p["preparation_seconds"],"promotion_eligible":False,"environment":{"gpu":torch.cuda.get_device_name(),"torch":torch.__version__},"artifacts":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"]=d.base.payload_sha256(report); d.base.write_json(output/"report.json",report); print(json.dumps(comparisons),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=["freeze","run"]); p.add_argument("--registration",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--spec-sha"); p.add_argument("--output",type=Path)
    a=p.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if a.command=="freeze": freeze(a.registration,a.spec_sha,a.source)
    else: run(a.registration,a.source,a.output)

if __name__=="__main__": main()
