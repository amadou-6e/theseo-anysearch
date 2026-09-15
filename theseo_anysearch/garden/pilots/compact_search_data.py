"""Fresh compact-search corpus and independent dense ridge readouts."""
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from ..compact import CompactEncoder
from . import compact_smoke as helpers

d = helpers.d
FAMILIES = ("random_field", "oblique_sheets", "sphere_shells", "box_shells")
COUNTS = {"train": 192, "probe": 384, "selection": 96, "development": 96}
BARS = {"occupied_iou": .60, "boundary_f1": .70, "clearance_nmae": .10, "recovery_nmae": .10}


def scene(gid, family, fraction):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(gid.encode()).digest()[:8], "little"))
    xyz = np.stack(np.meshgrid(*([np.arange(-24, 25)/8]*3), indexing="ij"))
    if family == "random_field": field = gaussian_filter(rng.normal(size=(49, 49, 49)), 1.5, mode="reflect")
    elif family == "oblique_sheets":
        normal = rng.normal(size=3); normal /= np.linalg.norm(normal)
        field = np.abs(np.sin(rng.uniform(4, 8)*(xyz*normal[:, None, None, None]).sum(0)+rng.uniform(-np.pi, np.pi)))
    elif family in ("sphere_shells", "box_shells"):
        fields = []
        for _ in range(2):
            center = rng.uniform(-1, 1, 3)[:, None, None, None]
            if family == "sphere_shells": fields.append(np.abs(np.sqrt(((xyz-center)**2).sum(0))-rng.uniform(.4, 1.5)))
            else: fields.append(np.abs((np.abs(xyz-center)-rng.uniform(.3, 1.2, 3)[:, None, None, None]).max(0)))
        field = np.minimum(*fields)
    else: raise ValueError("unknown family")
    return field <= np.quantile(field, fraction)


def data(splits=None):
    result = {}; seen = set()
    for split in (list(COUNTS) if splits is None else splits):
        inputs, hidden, targets, ids, hashes = [], [], [], [], []
        for i in range(COUNTS[split]):
            gid = f"compact-search-v1-{split}-{i:04d}"
            occ = scene(gid, FAMILIES[(i % 12)//3], [.08, .16, .28][i % 3])
            sha = hashlib.sha256(occ.tobytes()).hexdigest()
            if sha in seen: raise ValueError("duplicate parent")
            seen.add(sha); ids.append(gid); hashes.append(sha)
            t = d.base.compute_geometry_targets(occ, truncation=8.)
            maskrng = np.random.default_rng(int.from_bytes(hashlib.sha256((gid+"-mask").encode()).digest()[:8], "little"))
            inputs.append(helpers.prior.crop(occ, 33).copy())
            hidden.append(helpers.prior.crop(maskrng.random(occ.shape)<.2, 33).copy())
            targets.append(np.stack([helpers.prior.crop(v, 17).copy() for v in (occ, t.boundary, np.maximum(t.signed_distance/8, 0))]))
        result[split] = {"occupancy": torch.tensor(np.stack(inputs), dtype=torch.bool),
                         "hidden": torch.tensor(np.stack(hidden), dtype=torch.bool),
                         "targets": torch.tensor(np.stack(targets), dtype=torch.float32).flatten(2),
                         "ids": ids, "parents": hashes}
    return result


def identity(rows):
    return {s: {"ids": v["ids"], "parents": v["parents"],
                "arrays": {k: d.digest_tensor(v[k]) for k in ("occupancy", "hidden", "targets")}}
            for s, v in rows.items()}


def make_encoder(config, source, source_record):
    torch.manual_seed(389)
    backbone = d.base.make_encoder(0, torch.device("cuda"))
    backbone.load_state_dict(torch.load(source/source_record["path"], map_location="cuda", weights_only=True))
    if d.base.encoder_state_sha256(backbone) != source_record["state_hash"]: raise ValueError("source state mismatch")
    torch.manual_seed(389)
    return CompactEncoder(backbone, config["dimension"], config["mode"], joint=config["joint"]).cuda()


@torch.no_grad()
def vectors(model, rows, deadline):
    model.eval(); values = []
    for i in range(0, len(rows["ids"]), 4):
        d.check_deadline(deadline)
        occ = rows["occupancy"][i:i+4].to("cuda", dtype=torch.float32)
        hidden = rows["hidden"][i:i+4].cuda()[:, None]
        values.append(model(d.base.VoxelLevel.from_occupancy(occ, unknown_mask=hidden), hidden).cpu())
    return torch.cat(values)


def ridge_fit(vectors, targets, alpha=1.):
    if len(vectors) != len(targets) or vectors.ndim != 2 or alpha <= 0: raise ValueError("invalid ridge inputs")
    x = vectors.double(); y = targets.flatten(1).double()
    mean = x.mean(0); scale = x.std(0, unbiased=False).clamp_min(1e-5)
    x = torch.cat(((x-mean)/scale, x.new_ones(len(x), 1)), 1)
    penalty = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)*alpha; penalty[-1, -1] = 0
    coefficients = torch.linalg.solve(x.T@x + penalty, x.T@y)
    if not torch.isfinite(coefficients).all(): raise FloatingPointError("nonfinite ridge coefficients")
    return {"mean": mean, "scale": scale, "coefficients": coefficients, "shape": list(targets.shape[1:])}


def ridge_predict(readout, vectors):
    x = (vectors.double()-readout["mean"])/readout["scale"]
    x = torch.cat((x, x.new_ones(len(x), 1)), 1)
    return (x@readout["coefficients"]).reshape(len(x), *readout["shape"]).clamp(0, 1)


def masks(rows):
    hidden = helpers.prior.crop(rows["hidden"], 17).flatten(1)
    free = rows["targets"][:, 0] < .5
    return {"occupied_iou": hidden, "boundary_f1": hidden, "clearance_nmae": ~hidden&free, "recovery_nmae": hidden&free}


def select_thresholds(prediction, rows):
    result = {}; eligible = masks(rows)
    for channel, task in enumerate(d.base.TASKS[:2]):
        selected = eligible[task]
        if not selected.any(): raise ValueError("empty selection query pool")
        result[task] = d.threshold(prediction[:, channel][selected].cpu().numpy(), rows["targets"][:, channel][selected].numpy())
    return result


def evaluate(prediction, rows, thresholds):
    result = {}; eligible = masks(rows); prediction = prediction.cpu()
    for task in d.base.TASKS:
        channel = 0 if task == "occupied_iou" else 1 if task == "boundary_f1" else 2
        p = prediction[:, channel]; y = rows["targets"][:, channel]; valid = eligible[task]
        if not bool(valid.any()): raise ValueError("empty task query pool")
        if task in d.base.TASKS[:2]:
            yes = p >= thresholds[task]; truth = y > .5
            stats = torch.stack(((yes&truth&valid).sum(1), (yes&~truth&valid).sum(1), (~yes&truth&valid).sum(1)), 1).numpy()
        else:
            stats = torch.stack(((p-y).abs().mul(valid).sum(1), valid.sum(1)), 1).numpy()
        result[task] = {"score": d.base.score(stats, task), "statistics": stats.tolist(),
                        "eligible_counts": valid.sum(1).tolist(),
                        "families": {f: d.base.score(stats[[i for i in range(len(stats)) if (i%12)//3==j]], task) for j, f in enumerate(FAMILIES)}}
    return result


def rank(record):
    ratios = []
    for task, bar in BARS.items():
        score = record["selection"][task]["score"]
        ratios.append(score/bar if task in d.base.TASKS[:2] else bar/max(score, 1e-8))
    return (sum(x >= 1 for x in ratios), min(ratios), float(np.minimum(ratios, 2).mean()),
            -record["config"]["dimension"], -record["config"]["id"])
