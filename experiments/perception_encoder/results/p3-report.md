# P3 architecture feasibility report

Status: `completed`

This is a random-weight feasibility result, not an encoder-quality comparison. The
run used the pinned pilot specification at
`f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d` and code
`fb85c08e2052c6043703e882306328f744b4693e` on an NVIDIA GeForce RTX 3060 Ti
(8 GiB, driver 560.94, CUDA 12.6, PyTorch 2.13.0+cu126). Compilation was disabled
and all candidates used FP32, batch size 1, seed 0, 100 warmups, 500 measured
inferences, and 20 measured training steps per radius.

## Result

| Candidate | Radius | p50 ms | p95 ms | Inference MiB | Training MiB | Train examples/s | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| current dense | 8 | 1.214 | 1.403 | 24.0 | 38.9 | 138.3 | 0.27 |
| current dense | 16 | 1.187 | 1.432 | 31.3 | 58.1 | 180.7 | 1.71 |
| current dense | 32 | 4.654 | 5.775 | 98.0 | 200.8 | 52.5 | 12.23 |
| dense residual | 8 | 2.402 | 2.729 | 23.6 | 39.3 | 123.2 | 0.35 |
| dense residual | 16 | 2.517 | 2.920 | 37.2 | 73.2 | 128.6 | 2.29 |
| dense residual | 32 | 7.363 | 8.345 | 146.6 | 353.8 | 39.1 | 16.55 |
| tri-planar | 8 | 5.227 | 5.915 | 23.6 | 39.7 | 68.4 | 0.13 |
| tri-planar | 16 | 5.199 | 5.726 | 39.0 | 61.5 | 64.8 | 0.41 |
| tri-planar | 32 | 5.913 | 6.498 | 161.4 | 252.5 | 63.2 | 1.48 |
| shared pyramid | 8 | 2.619 | 2.840 | 23.8 | 40.0 | 132.7 | 0.35 |
| shared pyramid | 16 | 4.537 | 5.067 | 24.2 | 43.9 | 84.4 | 0.70 |
| shared pyramid | 32 | 6.444 | 6.832 | 24.6 | 47.7 | 61.5 | 1.05 |

All 12 dense cells completed without OOM or contract failure. The matched parameter
range is 1,129,814 to 1,186,465, a maximum/minimum ratio of 1.050. The optional
sparse residual candidate was skipped for all three radii because MinkowskiEngine is
not installed on this Python/device platform.

Decision: retain the current dense reference, Tiny dense residual, Tiny tri-planar,
and Tiny shared pyramid for P4. The P3 rule rejects a candidate only for output
failure, radius-32 OOM, or simultaneous training-memory and p95-latency regressions
above 50% without a new required capability. No dense candidate met a rejection
condition. P3 intentionally makes no quality claim.

## Reproducibility

- Integration base: `0b8476f90a5da45b060bd641daedb8f5468d0189`
- Resolved config SHA-256: `5a30dbbc2ee70a053db541c2b861165c01d00529603da19b7434ac4eff54a739`
- JSON payload SHA-256: `5ab2dafd27394893fdb8087d12db68d6078b07f838a33d9e00b4b94a71d55dbf`
- JSON file SHA-256: `beef67c8db640fb5933368c449dc53343aeed6a5a467583b92857dcfa39460a0`
- Trials: 12 completed, 0 failed, 3 optional-backend skips, cap 15
- Accelerator time: 0.00950 hours

Run with:

```powershell
.\.venv\Scripts\python.exe -m experiments.perception_encoder.p3_profile `
  --output experiments/perception_encoder/results/p3-report.json
```

Raw per-cell shapes, allocated and reserved memory, checkpoint bytes, parameter
counts, failure fields, and the machine-readable decision are in `p3-report.json`.
