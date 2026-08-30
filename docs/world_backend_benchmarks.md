# World backend parity and benchmark methodology

This baseline verifies the in-memory contracts introduced by issues #218 through #222. It is intentionally generic over `WorldRead` and `WorldMutation`, so the same exact snapshots cover flat backends, overlay-resolved `WorldState`, and scenario-v2 callbacks.

## Parity scope

`HashMapWorld` is the oracle. Exact results are compared with `ChunkedWorld` for point reads, bounded enumeration, rays, counts, set/update/remove operations, cubic and non-cubic extents, partial edge chunks, chunk boundaries, empty worlds, and sparse/dense logical chunks. Overlay parity materializes the resolved view in the oracle and checks base overrides, tombstones, overlay-only additions, block counts, rays, reset isolation, and preservation of the shared base. `WorldState` parity additionally compares observations, masks, rewards and breakdowns, collision flags, termination/truncation, and enumerated world snapshots. Deterministic multi-agent, heterogeneous-agent, shared trail-union, and reset behavior are covered as environment-level parity fixtures.

The scenario-v2 callback table is captured through point, two-pass region, and ray callbacks and compared value-for-value with the same generic `ReadSnapshot`. Injected backend failures must cross the callback boundary as an explicit status. Compiler tests inject incomplete builds, malformed manifests, tuple-index corruption, whole-pack checksum corruption, and short reads.

Rendering parity currently compares the sorted resolved world enumeration consumed by rendering and trajectory snapshots. Pixel-level rendering belongs to #22; persisted replay fixtures belong to #20.

The fault wrapper injects deterministic point, region, ray, set, update, and remove failures by call number. Invalid extents, regions, coordinates, rays, checked arithmetic overflow, candidate-index corruption, candidate query/result budget exhaustion, pack short reads, checksum corruption, cache eviction, pinned overcommit, and failed prefetch are direct fixtures.

## Benchmark workloads

The harness tests chunk edges 16, 32, and 64 over a non-cubic `130 × 97 × 65` extent, which creates partial edge chunks. Full runs include:

- empty geometry;
- deterministic isolated sparse voxels;
- clustered sparse geometry;
- a closed shell/surface;
- a dense block.

There is no checked-in STL fixture suitable for this backend microbenchmark. STL-derived geometry should be added when a stable fixture is coordinated with the ingest/replay work, rather than embedding a synthetic file and calling it realistic.

Each report records the fixed seed, operating system, architecture, build profile, processor identifier when available, warmup count, measured sample count, and operations per sample. Results contain minimum, p50, p95, p99, and maximum nanoseconds per operation. Warmups are executed separately and excluded from samples.

Measured operations are point reads, radius-2 and radius-8 regional reads, length-8 and length-32 rays, mutations, backend reset/repopulation, overlay-only environment reset, environment step controls, and full enumeration. Resident chunk count, conservative decoded/storage estimates, and overlay memory are reported separately.

Overlay memory is measured from the actual number and representation of episode-local overrides and tombstones. Encoded bytes are measured by the Python world-pack benchmark documented in `world_pack_format.md`; this in-memory Rust report leaves that field `null` rather than mixing measurements from separate processes. Disk-backed runs obtain decoded, pinned, and pinned-overcommit bytes directly from `world_cache_metrics()`. Operating-system file-cache bytes remain a separate, explicitly unavailable measurement: region reads benefit from the OS cache, but neither process RSS nor decoded-cache bytes are presented as an OS residency guarantee.

## Commands

CI-safe wiring smoke test:

```text
cargo test --lib verification::tests::benchmark_smoke_matrix_is_ci_safe_and_separates_future_memory_metrics
```

Short release baseline used for checked-in conclusions:

```text
cargo run --release --bin world-bench -- --baseline --output runtime/world-bench-baseline.json
```

Full local benchmark:

```text
cargo run --release --bin world-bench -- --output runtime/world-bench-full.json
```

`runtime/` is ignored by Git. For compiled packs, `pack_reads` separates cold faults from hot cache hits and the decoded/pinned counters describe process-owned memory. The operating-system file cache remains outside that measurement. The original in-memory warmup/measured split continues to represent process-local construction followed by hot queries.

## Revisit criteria

Keep flat chunk indexing unless representative workloads demonstrate at least one of these problems across repeated release runs:

- p95 regional or ray latency grows materially with total resident chunks rather than queried chunks;
- sparse-world decoded metadata dominates block storage;
- full enumeration or residency lookup becomes a measured training bottleneck;
- no flat chunk edge provides acceptable point, region, and reset trade-offs.

Only then should an octree or other hierarchy be evaluated against this same baseline.
