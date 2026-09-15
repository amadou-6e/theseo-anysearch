# Trajectory storage benchmark (#456)

This is an experimental Python/Rust storage harness, not a production format change.
It compares pretty JSON, compact JSON + Zstandard, typed Protobuf (with/without
compression), and fixed-width binary (with/without compression). Dependencies:
existing Protobuf plus optional `pip install -e '.[storage-benchmark]'`.

```powershell
python -m usage.benchmarks.trajectory_storage.benchmark --episodes 32 --repetitions 3 --inputs <trajectory.json> --output runtime/trajectory-storage-456
```

Three reproducible synthetic 4096-step cases cover single-agent, cumulative voxel
mutations, and four-agent traces. Input files add real cases; inputs are hashed and
never modified. JSON metadata/world references, unknown fields, presence/default
distinctions, 32-bit coordinates and 64-bit rewards/counts are preserved. NaN/Inf
are rejected as nonportable JSON. Sidecar world geometry is not duplicated or
included in trajectory size. The native probe validates and renders real world
references using the existing viewer's loader and scene helpers.

The fixed-width candidate contains a readable JSON manifest, a little-endian
header + scalar record array, uint64 event offsets, and a typed Protobuf event
sidecar. Its record is 44 bytes including a presence mask. This deliberately uses
the same typed variable-event schema instead of inventing another custom nested
format. It is **not** dependency-free. Compressed candidates decompress once at
load time; all candidates then support in-memory step indexing.

The `.proto` documents the schema. The harness builds its equivalent descriptors
through Protobuf's public API so running the benchmark needs no `protoc`. Any
production Python/Rust implementation should generate both bindings from one
schema rather than maintain descriptors in parallel.

Write timings include encode/compress/buffered filesystem write, not fsync.
Reads are OS-cached, not claimed cold-disk reads. Each load/memory measurement uses
a fresh subprocess retaining multiple independent decodes of the same episode.
OS peak RSS includes the interpreter and imports; baseline and after-load values
are reported separately. On Windows current working-set RSS is also measured.
Native codec and fully materialized Python-dictionary representations are both
tested. Worker wall time includes process setup and all queries; it is not viewer
startup time. Raw reports/artifacts stay ignored under `runtime/`.

Scrub timings measure step-record access separately from cumulative occupancy
overlay reconstruction (equivalent to Rust's scan-through-selected-step algorithm).
The separate native probe measures the actual Rust mutation resolver, regional
loading, meshing, sorting and egui CPU painting/tessellation. Neither probe measures
GPU presentation or full interactive UI/async-LOD frame latency.
No claim of O(1) complete scene reconstruction follows from O(1) record access.

Tests cover exact round trips, missing/default fields, unknown metadata, mutations,
multi-agent values, numeric boundaries, and invalid binary indexes. Before choosing
a production migration, validate a diverse many-episode corpus, direct typed/zero-copy
readers if desired, and interactive GPU/async-LOD behavior. See `RUST-RESULTS.md`
for the completed bounded comparison and recommendation. `RESULTS.md` preserves
the preliminary Python-only r2 evidence; the native comparison uses fresh r3 inputs.

## Native probe and readable exports

```powershell
cargo test --manifest-path usage/benchmarks/trajectory_storage/rust/Cargo.toml --release
cargo build --manifest-path usage/benchmarks/trajectory_storage/rust/Cargo.toml --release
python -m usage.benchmarks.trajectory_storage.rust_benchmark runtime/trajectory-storage-456/report.json --probe usage/benchmarks/trajectory_storage/rust/target/release/trajectory-storage-probe.exe --output runtime/trajectory-storage-456/rust-report.json --episodes 32 --repetitions 3
python -m usage.benchmarks.trajectory_storage.rust_summarize runtime/trajectory-storage-456/report.json runtime/trajectory-storage-456/rust-report.json
python -m usage.benchmarks.trajectory_storage.inspect runtime/trajectory-storage-456/synthetic-single/binary-zstd --format binary-zstd --step 100
```

The Rust crate generates bindings from the checked-in `.proto` with pure Rust
codegen (no external protoc). It includes the current production viewer source
so its record types and rendering helpers are not copied. All six codecs decode
through serde_json::Value into the same actual StepData vector. Timings therefore
include this prototype conversion cost, not an optimized direct typed or mmap
reader. Every worker checks exact cross-language equality against the original
JSON before materializing viewer records; the viewer's existing float32 fields
remain unchanged. Three fresh workers per format report cached first-record load,
retained working set/peak, indexed record queries and cumulative mutation scans.

The real-world scene probe is format-independent after decoding: it loads the
verified original artifact/world, initializes the viewer, and samples steps 0,
quarter, half and final at radius 16 using synchronous loading and CPU tessellation.
It excludes GPU upload/presentation, complete sidebar UI and async camera chunk
selection. Synthetic fixtures intentionally have no world pack and cannot supply
scene timings. Real artifact load failures are fatal, not silently skipped.
Source, input, report and executable hashes guard the measured run.

The inspect command exports readable JSON for an entire trajectory or one step.
`--output <new-file.json>` refuses to overwrite an existing file. Reports and
generated artifacts remain ignored under `runtime/`; crate build output is ignored.
