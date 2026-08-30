# Initial world backend benchmark results

This report records the first #218/#219 in-memory baseline. It does not make claims about persistence, encoded size, operating-system file caching, overlays, pinning, or prefetch.

## Conclusion

Chunk edge **32 remains the provisional default**. It is the middle trade-off: edge 16 creates more resident chunk metadata for sparse and boundary-heavy geometry, while edge 64 makes regional enumeration inspect larger per-chunk block maps. The release baseline should be rerun on representative training machines before changing the default.

On the initial Windows x86-64 release run (Intel Family 6 Model 183, fixed seed `37804723765785`, one warmup and five samples of 256 operations):

| Workload | Edge | Resident chunks | Point p50 (ns) | Radius-8 region p50 (ns) | Length-32 ray p50 (ns) |
|---|---:|---:|---:|---:|---:|
| Isolated sparse | 16 | 205 | 48.4 | 203.1 | 1,535.9 |
| Isolated sparse | 32 | 41 | 63.3 | 216.8 | 353.1 |
| Isolated sparse | 64 | 10 | 62.5 | 657.4 | 1,940.2 |
| Dense block | 16 | 8 | 46.5 | 156.2 | 2,050.0 |
| Dense block | 32 | 8 | 45.3 | 56.6 | 155.9 |
| Dense block | 64 | 1 | 64.1 | 23,071.5 | 894.1 |

Edge 32 sharply reduced sparse resident-chunk count versus edge 16, remained competitive on sparse point/region reads, and won the sparse ray control. It also won the dense point, region, and ray controls in this run. Edge 64's large per-chunk scan caused a severe dense and shell regional penalty. This supports 32 as a balanced default rather than a universal winner.

Backend reset p50 values were geometry-dependent: for the dense block, edges 16/32/64 measured approximately 3.28/1.92/1.29 ms; for isolated sparse geometry they measured 59/61/62 µs. These are current rebuild baselines and should be complemented by overlay-only reset measurements after #220 merges.

No octree or hierarchy is justified by the current flat-backend results. The criteria in [world_backend_benchmarks.md](world_backend_benchmarks.md) define when to revisit that decision.

## Reproduction

The numeric source report is generated locally and intentionally excluded from Git:

```text
cargo run --release --bin world-bench -- --baseline --output runtime/world-bench-baseline.json
```

The report includes machine/build metadata and per-distribution percentiles. This checked-in summary should be updated when persistent, overlay, and cache adapters land, without rewriting the benchmark harness.
