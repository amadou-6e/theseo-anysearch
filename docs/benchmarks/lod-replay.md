# Camera LOD replay benchmark

Run from `theseo_anysearch/core` with:

```text
cargo run --release --bin lod-replay-bench
```

The 60,000 x 40,000 x 20,000 fixture contains 256 indexed sparse chunks. The
camera is zoomed out across the complete index with a hard budget of 64 visible
chunks and 16 detailed chunks. Results recorded on Windows x86-64 on
31 August 2026:

| Measurement | Result |
| --- | ---: |
| Time to first coarse overview | 0.0372 ms |
| First detailed refinement | 2.8086 ms |
| Frame preparation p50 | 1.3258 ms |
| Frame preparation p95 | 1.3957 ms |
| Viewer RSS | 10,854,400 bytes |
| Indexed / considered chunks | 256 / 256 |
| Detailed / coarse chunks | 16 / 48 |
| Resident chunks | 16 |
| Pack reads | 16 |

The overview examined index metadata only. Detailed refinement decoded exactly
the 16-chunk detail budget; the other 48 visible chunks remained coarse, and
the remaining 192 indexed chunks were neither rendered nor decoded. Zooming
out therefore cannot turn into full-world voxel loading.
