# Initial world backend benchmark results

This report records the #218-#222 in-memory baseline after rebasing onto the overlay and scenario-query implementations. It does not make claims about disk residency, operating-system file caching, pinning, or prefetch.

## Conclusion

Chunk edge **32 remains the provisional default**. It is the middle trade-off: edge 16 creates more resident chunk metadata for sparse and boundary-heavy geometry, while edge 64 makes regional enumeration inspect larger per-chunk block maps. The release baseline should be rerun on representative training machines before changing the default.

On the post-overlay Windows x86-64 release run (Intel Family 6 Model 183, fixed seed `37804723765785`, one warmup and five samples of 256 operations):

| Workload | Edge | Resident chunks | Point p50 (ns) | Radius-8 region p50 (ns) | Length-32 ray p50 (ns) |
|---|---:|---:|---:|---:|---:|
| Isolated sparse | 16 | 205 | 43.0 | 174.2 | 1,322.7 |
| Isolated sparse | 32 | 41 | 94.5 | 314.5 | 611.3 |
| Isolated sparse | 64 | 10 | 78.5 | 888.3 | 2,555.1 |
| Dense block | 16 | 8 | 61.7 | 210.2 | 1,921.9 |
| Dense block | 32 | 8 | 69.9 | 100.0 | 309.0 |
| Dense block | 64 | 1 | 56.6 | 44,071.5 | 1,753.5 |

Edge 32 sharply reduced sparse resident-chunk count versus edge 16, remained competitive on sparse point/region reads, and won the sparse ray control. It also won the dense point, region, and ray controls in this run. Edge 64's large per-chunk scan caused a severe dense and shell regional penalty. This supports 32 as a balanced default rather than a universal winner.

Backend reset/repopulation remains geometry-dependent and intentionally separate from episode reset. Across the five workloads, median overlay-only environment reset p50 was approximately 0.4/0.3/0.3 µs for edges 16/32/64. Each fixture reported 640 bytes of episode-overlay storage before reset, and reset returned that storage to zero without rebuilding the shared base. The result validates #220's lifecycle design and shows that chunk edge does not materially affect overlay clearing.

No octree or hierarchy is justified by the current flat-backend results. The criteria in [world_backend_benchmarks.md](world_backend_benchmarks.md) define when to revisit that decision.

## Reproduction

The numeric source report is generated locally and intentionally excluded from Git:

```text
cargo run --release --bin world-bench -- --baseline --output runtime/world-bench-baseline.json
```

The report includes machine/build metadata and per-distribution percentiles. This checked-in summary should be updated when residency and cache adapters land, without rewriting the benchmark harness.
