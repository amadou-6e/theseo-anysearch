# Experimental fine729 encoder package

Issue #403 / PR #404. Fixed input33-cubed, central17-cubed geometry prediction;
normalized spatial code729 float32 numbers (2,916 bytes per example). The latent
layout is1x9x9x9, not a generic semantic global embedding. No RLlib dependency.

The three-seed replication passed every preregistered family target and code-
necessity check. [Replication assessment](perception-compact-replication-assessment.md)
contains the scores and limitations. Default seed401 is the lowest numeric ID,
not selected by development performance. Seeds402/403 are also exported.

## Use

Run from this experiment worktree with the project environment. The artifact
directory is local and intentionally untracked:
`C:/CodeWorkspace/theseo-anysearch/runtime/compact-package-v1-run1`.

```python
import torch
from theseo_anysearch.garden.compact_package import load_compact_package

encoder, head, metadata = load_compact_package(
    "C:/CodeWorkspace/theseo-anysearch/runtime/compact-package-v1-run1",
    seed=401,
    device="cuda",
)
# occupancy: binary (B,33,33,33); unknown_mask: matching boolean tensor.
code = encoder(occupancy, unknown_mask)  # detached float32 (B,729)
indices = torch.arange(17**3, device=code.device)[None].expand(len(code), -1)
raw = head(code, indices)               # (B,3,4913), gradients enabled for head
occupancy_probability = raw[:, 0].sigmoid()
boundary_probability = raw[:, 1].sigmoid()
normalized_distance = raw[:, 2].clamp(0, 1)
```

The loader checks the manifest digest, fixed interface contract, weight hashes
and restored state hashes. It returns a permanently eval/frozen encoder and a
separate native head whose parameters can be optimized. Use the raw outputs in
the appropriate training loss; the clamped/probability values are for evaluation.
The two distance metrics use this same distance channel on different free-cell
subsets. Central query indices follow tensor flatten order, not world coordinates.

Only full-validity stride1 binary occupancy with an explicit unknown mask is
supported. Move both inputs to the encoder device. Occupancy under unknown cells
is masked before the backbone. The interface does not expose raw backbone
features or its unused192-number projection. Calling encoder.train() keeps it
in evaluation mode; code extraction uses no_grad so head-only backprop works.

## Validation and limits

537 garden tests passed. Export equivalence passed for all three seeds; see the
[package report](perception-encoder-local-geometry/compact-package-report.json)
and [manifest](perception-encoder-local-geometry/compact-package-manifest.json).
The exporter checked64 development examples per seed
against saved native predictions, then takes a disposable head-only optimizer
step and requires the encoder state to remain unchanged. Modified heads are not
saved. Warm batch1 encoder latency on this RTX3060Ti was13.98ms median/14.77ms
p95 over50 synchronized repetitions; observed peak CUDA allocation54,901,248
bytes. This is a local measurement, not a hardware-independent performance claim.
The verification charged331.453s including300s test overhead. Report hash:
`786c803a8c648b283c5136d6332c95ba7a88b7a6179a51847fa4850ede8bf8d1`.
Weights are local only; Git contains the compact manifest/report.

Experimental synthetic local-geometry use only. Three seeds share one fresh
corpus and one backbone. Minimum family/seed F1 is .705934 against .70; this
is a narrow margin, not broad robustness. The common-head failure from #399
remains: quality with arbitrary heads is not established. No navigation,
topology, larger-field or promotion claim. PR review/merge is still required.
