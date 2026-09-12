"""Dense-output positive control for compact-code tiny-corpus learnability."""
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
from ..compact import CompactEncoder
from . import compact_learnability as previous

d = previous.d
PLAN = {"study_id": "compact-dense-v1", "sources": ["oracle", "table", "encoder"],
        "steps": 1024, "checkpoints": [128, 256, 512, 1024], "cap_seconds": 1200,
        "fit_cap_seconds": 600, "lr": .003, "generalization_claim": False}


def data(): return previous.data(PLAN["study_id"])


def scores(predictions, target):
    p, y = predictions.cpu().numpy(), target.cpu().numpy()
    a = previous.previous.metrics(p, y, .5, "occupied_iou")
    return {"iou": a["fixed05_score"], "f1": previous.previous.metrics(p, y, .5, "boundary_f1")["fixed05_score"],
            "auprc": a["auprc"], "bce": a["log_loss_nats"],
            "per_scene": [{"iou": previous.previous.metrics(p[i:i+1], y[i:i+1], .5, "occupied_iou")["fixed05_score"],
                           "f1": previous.previous.metrics(p[i:i+1], y[i:i+1], .5, "boundary_f1")["fixed05_score"]} for i in range(4)]}


def freeze(path, spec, source):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"): raise ValueError("commit source first")
    p = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
         "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-dense.md",
         "source_encoders": previous.previous.previous.prior.sources(source), "data": previous.identity(data()),
         "preparation_seconds": time.monotonic()-start}
    d.base.write_json(path, {"payload": p, "identity_sha256": d.base.payload_sha256(p)})


def run(path, source, output):
    start = time.monotonic(); env = json.loads(path.read_text()); p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or previous.previous.previous.prior.sources(source) != p["source_encoders"]: raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"): raise ValueError("source mismatch")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    rows=data()
    if previous.identity(rows) != p["data"]: raise ValueError("data mismatch")
    output.mkdir(parents=True, exist_ok=False)
    hidden=rows["hidden"].cuda()[:, None]; target=rows["targets"].cuda(); oracle=rows["codes"].float().cuda()
    level=d.base.VoxelLevel.from_occupancy(rows["occupancy"].cuda(), unknown_mask=hidden)
    weight=(1-target).sum()/target.sum(); deadline=start+PLAN["cap_seconds"]-p["preparation_seconds"]
    records=[]
    for kind in PLAN["sources"]:
        torch.manual_seed(386)
        if kind=="encoder":
            backbone=d.base.make_encoder(0, torch.device("cuda"))
            backbone.load_state_dict(torch.load(source/p["source_encoders"][0]["path"],map_location="cuda",weights_only=True))
            if d.base.encoder_state_sha256(backbone)!=p["source_encoders"][0]["state_hash"]: raise ValueError("backbone mismatch")
            torch.manual_seed(386); provider=CompactEncoder(backbone,128,"grid",joint=True).cuda()
        elif kind=="table": provider=nn.Embedding(4,128).cuda()
        else: provider=None
        decoder=nn.Linear(6 if kind=="oracle" else 128,4913).cuda()
        parameters=list(decoder.parameters())+([x for x in provider.parameters() if x.requires_grad] if provider is not None else [])
        opt=torch.optim.AdamW(parameters,lr=.003,weight_decay=.01)
        begin=time.monotonic(); stop=min(deadline,begin+600); curve=[]; torch.cuda.reset_peak_memory_stats()
        for step in range(1024):
            d.check_deadline(stop)
            code=previous.code_for(kind,provider,level,hidden,oracle)
            logits=decoder(code); loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=weight)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step+1 in PLAN["checkpoints"]:
                with torch.no_grad(): pred=decoder(previous.code_for(kind,provider,level,hidden,oracle)).sigmoid()
                value=scores(pred,target); curve.append({"step":step+1,"loss":float(loss.detach()),**value})
                print(json.dumps({"kind":kind,"step":step+1,"iou":value["iou"],"f1":value["f1"]}),flush=True)
        torch.cuda.synchronize(); d.check_deadline(stop)
        records.append({"name":kind,"curve":curve,"seconds":time.monotonic()-begin,
                        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"parameters":sum(x.numel() for x in parameters),
                        "overfit_target_met":curve[-1]["iou"]>=.95 and curve[-1]["f1"]>=.97})
        torch.save({"provider":provider.state_dict() if provider is not None else None,"decoder":decoder.state_dict(),"predictions":pred.cpu()},output/(kind+".pt"))
        del provider,decoder,opt,parameters,code,logits,loss,pred
    report={"registration":env,"status":"dense_learnability_completed","records":records,
            "elapsed_seconds":time.monotonic()-start+p["preparation_seconds"],"promotion_eligible":False,
            "environment":{"gpu":torch.cuda.get_device_name(),"torch":torch.__version__},
            "artifacts":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"]=d.base.payload_sha256(report); d.base.write_json(output/"report.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=["freeze","run"])
    parser.add_argument("--registration",type=Path,required=True); parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--output",type=Path); a=parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if a.command=="freeze": freeze(a.registration,a.spec_sha,a.source)
    else: run(a.registration,a.source,a.output)


if __name__=="__main__": main()
