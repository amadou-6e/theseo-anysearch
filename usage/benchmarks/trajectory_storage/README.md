# Trajectory storage benchmark (#456)

This is an experimental Python storage harness, not a production format change.
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
included in trajectory size; this harness does not render those references.

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
They do **not** measure actual Rust/egui scrubbing, chunk loading, meshing or frame
presentation. Those are required follow-up measurements before a format decision.
No claim of O(1) complete scene reconstruction follows from O(1) record access.

Tests cover exact round trips, missing/default fields, unknown metadata, mutations,
multi-agent values, numeric boundaries, and invalid binary indexes. Before choosing
a production format, add Rust reader/startup/actual scrub measurements, a diverse
many-episode corpus, repeated load workers and codec/schema implementation costs.
