# Frozen collision readout usage

This is an experimental small-field head, not a certified collision checker.
Read the [assessment](perception-encoder-local-geometry/compact-collision-readout-assessment.md)
and completed verification before interpreting its scores. Use the issue407
worktree and repository virtual environment until the code is reviewed/integrated.
No model weights or generated corpora are committed.

The native variant is a composition: frozen729 code -> frozen geometry decoder ->
new trainable collision head. Its three predicted geometry channels are not raw
occupancy or ground-truth labels. Other variants consume the729 code directly.
All query coordinates below are central17 coordinates in0..16; subtract8 from
input33 voxel indices. Each path has9 padded nodes and a boolean valid-node mask,
with a visibly free starting node and2/3/5/9 actual six-connected nodes.

```python
import json
from pathlib import Path
import torch
from theseo_anysearch.garden.compact_package import load_compact_package, file_sha
from theseo_anysearch.garden.collision_readout import NativeCollisionHead
from theseo_anysearch.garden.pilots.compact_replication import configure_cuda

configure_cuda()
runtime = Path("C:/CodeWorkspace/theseo-anysearch/runtime")
run = runtime / "compact-collision-readout-v1-run1"
report = json.loads((run / "report.json").read_text())
verification = json.loads((run / "verification.json").read_text())
assert verification["verified"]
assert verification["report_payload_sha256"] == report["report_payload_sha256"]

# Seed401 is a numeric example, not a test-selected winning seed.
stage = report["stages"]["head-401-native"]
path = run / stage["checkpoint"]
assert file_sha(path) == report["artifacts"][stage["checkpoint"]]
saved = torch.load(path, map_location="cpu", weights_only=True)
encoder, decoder, _ = load_compact_package(
    runtime / "compact-package-v1-run1", seed=401, device="cuda"
)
decoder.eval().requires_grad_(False)
head = NativeCollisionHead().cuda().eval()
head.load_state_dict(saved["model"])

# Supply occupancy/unknown: (B,33,33,33), binary/bool on the same device.
# paths: (Q,9,3) int64; valid: (Q,9) bool; geometry_indices: (Q,) int64.
with torch.no_grad():
    code = encoder(occupancy.cuda(), unknown.cuda())
    decoded = []
    for first in range(0, len(code), 8):
        z = code[first:first + 8]
        indices = torch.arange(4913, device="cuda")[None].expand(len(z), -1)
        logits = decoder(z, indices)
        decoded.append(torch.cat((logits[:, :2].sigmoid(),
                                  logits[:, 2:].clamp(0, 1)), dim=1))
    features = torch.cat(decoded).reshape(-1, 3, 17, 17, 17)
    probability = head(features, paths.cuda(), valid.cuda(),
                       geometry_indices.cuda()).sigmoid()
```

The selection lock contains separate `pooled` and `uncertain` thresholds under
`thresholds["401-native"]`. Never choose between them using ground-truth collision
labels. The uncertain threshold was evaluated on paths with unknown voxels and
no visible occupied voxel. Identifying that subset requires observed map state;
it is not information supplied by the729-code-only interface.

The observability-assisted diagnostic checks visible occupancy directly on definite
paths and uses learned predictions only for unresolved paths. Keep it distinct
from pure-head results. A prediction of no collision in hidden space is not proof
that the path is safe, and the empirical calibration recall is not a guarantee.
The experiment does not validate noisy sensors, contiguous occlusion, arbitrary
path lengths, nonzero body radius, larger fields, or real navigation.

For the `resize_conv` variant, load `head-401-resize_conv` into
`UpsamplingCollisionHead("resize_conv")` and pass the encoder code reshaped to
`(B,1,9,9,9)` directly. Do not run the native decoder for that variant. A new head
fit or selection comparison needs a fresh declared protocol/split; do not tune
on this study's already-used test sets or mutate the packaged normalization.
