# Chunk render-cache benchmark

Run from `theseo_anysearch/core` with:

```text
cargo run --release --bin render-cache-bench
```

The benchmark uses a fully occupied 32 x 32 x 32 chunk and 200 unchanged-frame
cache lookups. Results recorded on 31 August 2026 on Windows x86-64:

| Measurement | Result |
| --- | ---: |
| Occupied voxels | 32,768 |
| Naive independent-voxel faces | 196,608 |
| Exposed faces | 6,144 |
| Emitted triangles | 12,288 |
| Face reduction | 96.875% |
| Cold surface build | 17.3773 ms |
| Cached frame geometry preparation p50 | 0.0003 ms |
| Cached frame geometry preparation p95 | 0.0007 ms |
| Cache builds / hits | 1 / 200 |
| Cache hit rate | 99.5025% |

## Greedy-meshing decision

Greedy coplanar merging is not justified for the current agent-local CPU viewer.
Simple exposed-face extraction already removes 96.875% of the naive dense
geometry, a cold chunk builds within roughly one 60 Hz frame, and unchanged
frame preparation is negligible. Reconsider greedy meshing after camera-frustum
LOD benchmarks demonstrate that simultaneously visible detailed chunks make
the remaining 6,144 faces per dense chunk a material frame-time bottleneck.

This microbenchmark measures CPU surface construction and cached geometry
preparation. It deliberately excludes GPU submission because the current replay
viewer uses the egui CPU painter.
