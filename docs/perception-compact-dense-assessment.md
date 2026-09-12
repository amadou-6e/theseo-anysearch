# Compact dense-output positive control

Issue #386, source2afe8f0, specfd8eba978a6100b997cc762dbea6078d3b0f5686.
Run compact-dense-v1-run1. Disposition retain engineering direction, not promotion.

|Source|Final IoU|Final F1|1024-update seconds|Overfit target|
|---|---:|---:|---:|---|
|Oracle6|.9223|.9596|1.172|Not met|
|Learned table128|1.000|1.000|1.250|Met|
|Actual voxel encoder128|1.000|1.000|72.422|Met|

The table and actual encoder already reached perfect scores at128 updates and
retained them at1024. This establishes a learnable compact128 training path on
four scenes when a dense linear output head replaces coordinate-query decoding.
It does NOT establish generalization or useful hidden-cell perception: all central
voxels on those same scenes were training labels. Oracle's .922 result remains
below the frozen target despite theoretical rank sufficiency; do not erase it.

Decoder structure, exposure and optimizer LR differ from #385, so this is a positive
control, not an isolated causal decoder ablation. The large dense head is a training
and diagnostic readout; no claim of compact deployment head parameter efficiency.
All information from the actual voxel input passes through128 numbers; no local
feature skip or generator code enters that path.

Next action: fresh multi-scene dense-reconstruction training with independent frozen
readouts, followed by the approved budgeted optimization campaign. Do not select
architectures from tiny-corpus scores. User explicitly authorized up to48 GPU-hours
on2026-09-12; merges still require review. Keep budget usage and run checkpoints in
the campaign ledger and issues. No stop at this intermediate milestone.

Elapsed76.765s including preparation.449 garden tests passed. Three checkpoint and
full-prediction artifacts retained outside Git, with hashes. Registration/report:
docs/perception-encoder-local-geometry/compact-dense-{preregistration,report}.json.
Payloadab01e6a66c5c83969149f8ba9bbfdd2df3eff28f7b2b25f4573b7b5f19939e96.
No active process remains for this run; no integration merges or promotion.
