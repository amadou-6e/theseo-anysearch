# Experimental frozen-code collision head

Issue405 / PR406. See the [study assessment](perception-encoder-local-geometry/compact-validation-assessment.md)
before using any result. This is a local synthetic-data experiment, not a certified
collision checker, planner, or substitute for checking visible occupied voxels.
The observed input is33 cubed; candidate paths are confined to its central17 cube.
Collision means an occupied discrete path node for a zero-radius agent. It does
not cover swept volumes between arbitrary points, finite body radius, or dynamics.

After the run completes, the runtime artifact directory contains the12 fitted
heads and the calibration-only threshold lock. The snippet below loads seed401
by numeric ID, not because it was the best test seed. Code729 is a1x9x9x9 spatial
layout, not a192-dimensional global vector. No raw/spatial bypass enters this head.

```python
import json
from pathlib import Path
import torch
from theseo_anysearch.garden.compact_package import load_compact_package, file_sha
from theseo_anysearch.garden.collision_transfer import CollisionHead

runtime = Path("C:/CodeWorkspace/theseo-anysearch/runtime")
run = runtime / "compact-collision-transfer-v1-run1"
report = json.loads((run / "report.json").read_text())
stage = report["stages"]["head-401-compact"]
checkpoint_path = run / stage["checkpoint"]
assert file_sha(checkpoint_path) == report["artifacts"][stage["checkpoint"]]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
encoder, native_geometry_head, manifest = load_compact_package(
    runtime / "compact-package-v1-run1", seed=401, device="cuda"
)
del native_geometry_head
head = CollisionHead(1).cuda().eval()
head.load_state_dict(checkpoint["model"])
lock = json.loads((run / "selection-lock.json").read_text())
threshold = lock["thresholds"]["401-compact"]

# Supply real binary occupancy/boolean unknown masks, both shape (B,33,33,33).
# Paths: (Q,9,3) int64 coordinates in [0,16], central-crop coordinates.
# Valid: (Q,9) bool. Pad with the endpoint; mark padded nodes invalid.
# Geometry indices: (Q,) int64 mapping each path to its input geometry.
with torch.no_grad():
    code = encoder(occupancy.cuda(), unknown.cuda())
    collision_probability = head(
        code.reshape(-1, 1, 9, 9, 9), paths.cuda(), valid.cuda(),
        geometry_indices.cuda()
    ).sigmoid()
    predicted_collision = collision_probability >= threshold
```

The trained path distribution uses simple six-connected walks of2/3/5/9 nodes,
with a visibly known-free starting voxel. Inputs outside that distribution are
not validated by this study. A predicted non-collision is not proof of safety in
unobserved space. The report's `false_safe` is missed collisions divided by true
collisions, **not** the collision fraction among paths predicted safe.

For training another head, keep the encoder frozen and place only the new head's
parameters in the optimizer. The package encoder returns detached normalized
codes; its exported normalization must not be refitted on test data. Any new
training comparison needs fresh protocol/run identities and a declared selection
split, rather than tuning on these now-used test splits.
