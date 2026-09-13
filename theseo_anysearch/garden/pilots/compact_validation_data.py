"""Fresh paired stress scenes and geometry-disjoint short collision queries."""
import hashlib

import numpy as np
import torch

from . import compact_diversity_data as prior

SEEN = prior.old.FAMILIES
UNSEEN = ("solid_ellipsoids", "capsule_forest", "gyroid_walls")
FAMILIES = SEEN + UNSEEN
CONDITIONS = [("baseline", .16, .2, 0., False)]
CONDITIONS += [(f"density-{n}", n, .2, 0., False) for n in (.03, .08, .28, .45)]
CONDITIONS += [(f"missing-{n}", .16, n, 0., False) for n in (0., .1, .4, .6)]
CONDITIONS += [(f"noise-{n}", .16, .2, n, False) for n in (.01, .05)]
CONDITIONS += [("slab", .16, .2, 0., True)]
COUNTS = {"train": 1536, "calibration": 384, "test": 384, "ood": 288}
crop = prior.old.helpers.prior.crop


def scene(gid, family, fraction):
    if family in SEEN:
        return prior.scene(gid, family, fraction)
    rng = np.random.default_rng(prior.seed(gid))
    xyz = np.stack(np.meshgrid(*([np.arange(-24, 25) / 8] * 3), indexing="ij"))
    if family == "solid_ellipsoids":
        fields = []
        for _ in range(5):
            center = rng.uniform(-1.5, 1.5, 3)[:, None, None, None]
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            local = np.einsum("ij,jxyz->ixyz", rotation, xyz - center)
            fields.append(((local / rng.uniform(.3, 1.3, 3)[:, None, None, None]) ** 2).sum(0))
        field = np.minimum.reduce(fields)
    elif family == "capsule_forest":
        fields = []
        for _ in range(6):
            a, b = rng.uniform(-2, 2, (2, 3)); vector = b - a
            relative = xyz - a[:, None, None, None]
            t = np.clip((relative * vector[:, None, None, None]).sum(0) / np.dot(vector, vector), 0, 1)
            fields.append(np.sqrt(((relative - vector[:, None, None, None] * t) ** 2).sum(0)))
        field = np.minimum.reduce(fields)
    elif family == "gyroid_walls":
        x, y, z = xyz * rng.uniform(2, 5) + rng.uniform(-np.pi, np.pi, 3)[:, None, None, None]
        field = np.abs(np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x))
    else:
        raise ValueError("unknown generator")
    return field <= np.quantile(crop(field, 17), fraction)


def observations(gid, occupancy, missing=.2, noise=0., slab=False):
    rng = np.random.default_rng(prior.seed(gid + "-observation"))
    uniforms = rng.random((33, 33, 33)); flips = rng.random((33, 33, 33))
    mask = uniforms < missing
    if slab:
        mask[:] = False; axis = int(rng.integers(3)); start = int(rng.integers(27))
        slices = [slice(None)] * 3; slices[axis] = slice(start, start + 7)
        mask[tuple(slices)] = True
    values = occupancy.copy()
    values[(flips < noise) & ~mask] ^= True
    return values, mask


def robustness(count=24):
    result = {}
    for condition, fraction, missing, noise, slab in CONDITIONS:
        for family in FAMILIES:
            rows = {k: [] for k in ("occupancy", "hidden", "targets", "ids", "parents", "actual_density")}
            for i in range(count):
                gid = f"compact-robustness-v1-{family}-{i:03d}"
                parent = scene(gid, family, fraction)
                truth = prior.d.base.compute_geometry_targets(parent, truncation=8.)
                values, hidden = observations(gid, crop(parent, 33), missing, noise, slab)
                rows["occupancy"].append(values); rows["hidden"].append(hidden)
                rows["targets"].append(np.stack([crop(v, 17) for v in (parent, truth.boundary, np.maximum(truth.signed_distance / 8, 0))]))
                rows["ids"].append(gid); rows["parents"].append(hashlib.sha256(parent.tobytes()).hexdigest())
                rows["actual_density"].append(float(crop(parent, 17).mean()))
            rows.update(condition=condition, family=family, density=fraction)
            for name in ("occupancy", "hidden", "targets"):
                rows[name] = torch.from_numpy(np.stack(rows[name])).to(torch.float32 if name == "targets" else torch.bool)
            rows["targets"] = rows["targets"].flatten(2)
            result[f"{condition}/{family}"] = rows
    return result


def paths(gid, occupancy, hidden, count=32):
    if occupancy.shape != (17, 17, 17) or hidden.shape != occupancy.shape:
        raise ValueError("central17 path geometry required")
    rng = np.random.default_rng(prior.seed(gid + "-paths"))
    starts = np.argwhere(~occupancy & ~hidden)
    if not len(starts):
        raise ValueError("no visibly free start")
    offsets = np.concatenate((np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)))
    result, valid = [], []
    for i in range(count):
        length = (2, 3, 5, 9)[i % 4]
        for _ in range(2048):
            walk = [starts[rng.integers(len(starts))]]
            while len(walk) < length:
                candidates = [p for p in walk[-1] + offsets if np.all((p >= 0) & (p < 17)) and not any(np.array_equal(p, q) for q in walk)]
                if not candidates: break
                walk.append(candidates[rng.integers(len(candidates))])
            if len(walk) == length: break
        else:
            raise ValueError("path sampling cap exhausted")
        result.append(np.stack(walk + [walk[-1]] * (9 - length)))
        valid.append(np.arange(9) < length)
    return np.stack(result), np.stack(valid)


def collision_data(counts=None):
    counts = COUNTS if counts is None else counts
    result = {}; seen = set()
    for split, count in counts.items():
        families = UNSEEN if split == "ood" else SEEN
        rows = {k: [] for k in ("occupancy", "hidden", "paths", "valid", "labels", "visible_hit", "unknown_path",
                                "ids", "parents", "families", "densities")}
        for i in range(count):
            family = families[(i // 3) % len(families)]; fraction = (.08, .16, .28)[i % 3]
            gid = f"compact-collision-transfer-v1-{split}-{i:04d}"
            parent = scene(gid, family, fraction); digest = hashlib.sha256(parent.tobytes()).hexdigest()
            if digest in seen: raise ValueError("duplicate collision parent")
            seen.add(digest)
            values, hidden = observations(gid, crop(parent, 33))
            central = crop(parent, 17); central_hidden = crop(hidden, 17)
            query, valid = paths(gid, central, central_hidden)
            x, y, z = query.transpose(2, 0, 1)
            occupied = central[x, y, z]; unknown = central_hidden[x, y, z]
            for key, value in {"occupancy": values, "hidden": hidden, "paths": query, "valid": valid,
                               "labels": (occupied & valid).any(1), "visible_hit": (occupied & ~unknown & valid).any(1),
                               "unknown_path": (unknown & valid).any(1)}.items(): rows[key].append(value)
            rows["ids"].append(gid); rows["parents"].append(digest); rows["families"].append(family); rows["densities"].append(str(fraction))
        for key in ("occupancy", "hidden", "paths", "valid", "labels", "visible_hit", "unknown_path"):
            rows[key] = torch.from_numpy(np.stack(rows[key]))
        for family in families:
            labels = rows["labels"][[f == family for f in rows["families"]]]
            if min(int(labels.sum()), int((~labels).sum())) < 20: raise ValueError("insufficient collision class support")
        result[split] = rows
    return result


def identity(rows):
    return {split: {k: prior.d.digest_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in row.items()} for split, row in rows.items()}
