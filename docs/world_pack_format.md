# Compiled world packs

Large finite worlds are compiled before training into a content-addressed directory. The identity hashes the source content, finite extent, coordinate/schema contract, and compiler settings; it never includes a machine-local source path. This lets Tune trials with identical inputs reuse one artifact.

Each completed entry contains:

- `manifest.json`: versioned world contract, compiler settings, pack checksum, and per-chunk integrity metadata.
- `index.json`: tuple-coordinate keys (`x,y,z`) mapped to byte ranges in the pack.
- `world.pack`: concatenated, independently decodable non-background chunks.
- `candidates.idx`: versioned coarse-region/kind ranges bound to the world identity.
- `candidates.bin`: fixed-width public-coordinate spawn, goal, surface, and portal records.
- `COMPLETE`: written last and containing the world identity. Readers reject entries without it.

The compiler selects among a constant-size uniform representation, sorted sparse `u32` indices, and a zlib-compressed dense bitset. Empty chunks are omitted. Selection compares the actual encoded lengths, with `sparse_max_fraction` providing an explicit tuning boundary. Boxes are intersected directly with chunks, and `.npy` pool grids are memory-mapped and visited chunk-by-chunk; neither path creates a full-world list of Python coordinate tuples.

Publication uses the shared heartbeat/token cache lock. A build is written to a unique temporary directory, fully validated, and atomically renamed. Corrupt entries are rebuilt when their source still exists. When only a pack identity is available, corruption raises an explicit `WorldPackUnavailableError` because rebuilding is impossible.

## Runtime residency

Compiled packs can be attached to Gymnasium and PettingZoo workers without
materializing the world:

```yaml
env:
  geometry:
    compiled_world_path: runtime/worlds/<identity>
    maximum_decoded_bytes: 268435456
    prefetch_margin: 2
```

Each process shares one immutable backend between environment clones. Chunks
are region-read from `world.pack`, decoded into a bounded LRU, and pinned for
the union of the agents' movement and observation envelopes while a step is
executing. Reset may synchronously establish initial residency. Subsequent
envelopes are prefetched on Rust worker threads and joined before the next
step. A custom outcome that teleports or mutates outside the predicted
envelope takes a synchronous correctness fallback; it may make that step cold,
but the mutation is not committed until the target chunk has loaded and the
coordinate has been validated.

`world_cache_metrics()` reports cache hits/misses, pack reads, evictions,
decoded and pinned bytes, resident and pinned chunks, and pinned overcommit.
The single-agent Gymnasium wrapper also publishes these counters under
`info["world_cache"]`. `decoded_bytes` is owned decoded memory. It deliberately
does not claim to measure bytes retained by the operating-system file cache.

Use `stage_compiled_world()` to copy a validated pack into a node-local,
content-addressed cache. Publication uses the same heartbeat-protected lock and
atomic rename protocol as compilation, so concurrent RLlib workers reuse one
complete staged entry.

Spawn and goal candidates are free cells adjacent to compiled geometry; surface
candidates are the corresponding occupied boundary cells. Their quality score
records local openness. Portal candidates require explicit semantic annotation
and are therefore empty for occupancy-only sources. Candidate buckets are read
lazily and sampled deterministically from world identity, seed, and stream;
cache population and worker scheduling do not affect selection. Waypoint routes
remain task/curriculum inputs and can be varied without recompiling immutable
geometry.

## Measuring encoding choices

`benchmark_encodings()` records source voxel count, occupied count, encoded bytes, compression ratio against a one-byte dense grid, and median encode/decode throughput. Benchmark representative uniform, sparse, and dense chunks at the intended chunk shapes (16, 32, and 64). The broader reproducible benchmark matrix belongs to issue #226; the format deliberately exposes encoding and decoded length so those measurements remain reproducible.

An initial Windows x86-64 run on 32³ chunks (20 repetitions, NumPy 2.4.3, CPython 3.14.4) produced the following smoke-benchmark results. Throughput is machine-specific; encoded sizes and selected encodings are deterministic.

| Occupancy | Encoding | Bytes | Ratio vs. uint8 | Encode voxels/s | Decode voxels/s |
|---:|---|---:|---:|---:|---:|
| 128 / 32,768 | dense zlib | 286 | 114.6× | 218M | 378M |
| 16,332 / 32,768 | dense zlib | 4,112 | 8.0× | 164M | 1,052M |
| 32,768 / 32,768 | uniform | 5 | 6,553.6× | 7,992M | 7,047M |
