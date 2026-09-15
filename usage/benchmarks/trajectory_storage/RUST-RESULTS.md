# Native replay storage measurements (#456)

Python report SHA256: `31bc96632e87809cadcf400e748edcd5ea49e65d4c3ef8b183d2bee1ce0512c4`.

Rust report SHA256: `0aee29a899cd6752f2d24f51160717e1dfa2d6f96a8d760d639994547f702edc`.

Probe SHA256: `a250b1db2ab838006edb00f4ca795227955dfa9b28752760c8f97432f03c51d6`; rustc 1.98.0 (88d9e12ae 2026-08-18).

Windows-10-10.0.19045-SP0; 3 fresh release workers per format; 32 independently decoded copies retained per worker.

Cached reads, buffered writes without fsync. Load includes prototype decoding through serde_json::Value and conversion to the actual viewer StepData vector. RSS is total working set, not allocation accounting. Scrub record access and actual cumulative mutation reconstruction are separate from CPU scene work. GPU presentation and full interactive/async-LOD latency are not measured.

## synthetic-single (4096 steps)

| Format | KiB | Write ms | Rust load ms | Retained RSS MiB | Record p95 us | Final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1112.84 | 37.97 | 10.75 | 23.1 | 0.10 | 0.014 |
| json-zstd | 96.57 | 9.83 | 10.59 | 23.1 | 0.10 | 0.013 |
| protobuf | 127.82 | 61.07 | 9.33 | 23.2 | 0.10 | 0.013 |
| protobuf-zstd | 71.21 | 61.49 | 9.40 | 23.3 | 0.20 | 0.014 |
| binary | 220.75 | 49.17 | 13.63 | 24.6 | 0.20 | 0.013 |
| binary-zstd | 80.46 | 54.45 | 14.75 | 24.8 | 0.10 | 0.013 |

## synthetic-mutations (4096 steps)

| Format | KiB | Write ms | Rust load ms | Retained RSS MiB | Record p95 us | Final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1174.28 | 41.39 | 11.09 | 23.6 | 0.20 | 0.022 |
| json-zstd | 97.46 | 10.69 | 11.65 | 23.9 | 0.20 | 0.022 |
| protobuf | 132.89 | 64.58 | 10.02 | 24.0 | 0.10 | 0.022 |
| protobuf-zstd | 71.91 | 67.31 | 10.46 | 24.5 | 0.10 | 0.022 |
| binary | 227.75 | 51.39 | 15.10 | 25.1 | 0.20 | 0.023 |
| binary-zstd | 75.28 | 53.46 | 14.74 | 25.4 | 0.20 | 0.023 |

## synthetic-multi (4096 steps)

| Format | KiB | Write ms | Rust load ms | Retained RSS MiB | Record p95 us | Final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 4313.69 | 158.03 | 30.09 | 43.7 | 0.20 | 0.031 |
| json-zstd | 324.04 | 33.82 | 29.73 | 43.8 | 0.20 | 0.032 |
| protobuf | 556.89 | 115.39 | 28.17 | 45.6 | 0.20 | 0.032 |
| protobuf-zstd | 263.12 | 110.52 | 28.99 | 45.7 | 0.20 | 0.032 |
| binary | 647.75 | 133.53 | 35.65 | 46.3 | 0.20 | 0.032 |
| binary-zstd | 257.85 | 120.81 | 38.44 | 46.3 | 0.20 | 0.033 |

## real-0-iter_000300 (4096 steps)

| Format | KiB | Write ms | Rust load ms | Retained RSS MiB | Record p95 us | Final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1034.91 | 38.14 | 10.88 | 24.5 | 0.20 | 0.018 |
| json-zstd | 8.29 | 6.63 | 10.44 | 24.5 | 0.10 | 0.019 |
| protobuf | 76.46 | 61.78 | 9.44 | 23.3 | 0.10 | 0.018 |
| protobuf-zstd | 9.71 | 59.48 | 9.08 | 23.2 | 0.10 | 0.020 |
| binary | 221.04 | 51.52 | 13.14 | 24.5 | 0.10 | 0.017 |
| binary-zstd | 16.18 | 59.24 | 13.17 | 24.7 | 0.10 | 0.019 |

Format-independent actual-world scene probe (pooled across formats/repeats): original-artifact load 20.12 ms; viewer world setup 9.69 ms; regional overlay/load/mesh/sort 0.01 ms; CPU paint/tessellate 0.03 ms. Selected regions contain 0-6 faces. Four selected steps, synchronous radius 16, actual scene helpers; not a dense-wall stress test or an end-to-end GUI startup/frame benchmark.

## real-1-iter_000100 (2 steps)

| Format | KiB | Write ms | Rust load ms | Retained RSS MiB | Record p95 us | Final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1.31 | 1.05 | 0.25 | 3.4 | 0.10 | 0.000 |
| json-zstd | 0.44 | 0.75 | 0.27 | 3.5 | 0.10 | 0.000 |
| protobuf | 0.63 | 0.94 | 0.20 | 3.4 | 0.10 | 0.000 |
| protobuf-zstd | 0.42 | 1.24 | 0.30 | 3.5 | 0.10 | 0.000 |
| binary | 1.11 | 2.45 | 0.72 | 3.4 | 0.10 | 0.000 |
| binary-zstd | 1.09 | 2.42 | 0.63 | 3.5 | 0.10 | 0.000 |

Format-independent actual-world scene probe (pooled across formats/repeats): original-artifact load 18.01 ms; viewer world setup 9.46 ms; regional overlay/load/mesh/sort 0.00 ms; CPU paint/tessellate 0.03 ms. Selected regions contain 0-0 faces. Four selected steps, synchronous radius 16, actual scene helpers; not a dense-wall stress test or an end-to-end GUI startup/frame benchmark.

## Assessment and disposition

Retain the benchmark; do not migrate production storage in this PR. Prefer
compact JSON + Zstd for a first archive-format pilot on the current single-episode
viewer workload, with decompression once at load and readable inspect/export.
The real long trace falls from 1034.91 KiB to 8.29 KiB; prototype writes take
6.63 ms versus 59.48 ms for compressed Protobuf and 59.24 ms for compressed
indexed binary. Its Rust load is 10.44 ms versus 9.08 and 13.17 ms respectively.
These modest cached-load differences are not claims of statistical superiority.

There is no scrub advantage from indexed binary once all candidates become the
same viewer vector: record access is timer-resolution-scale, while overlays still
scan prior steps. Whole-file compression does not obstruct in-memory scrubbing.
The Python native binary/Protobuf memory savings in phase 1 do not carry over as
a large advantage after Rust viewer materialization. A direct typed reader or mmap
integration could change that conclusion, but was not implemented here.

Synthetic multi-agent/high-entropy traces favor compressed Protobuf or indexed
binary for size (263.12 / 257.85 KiB versus JSON's 324.04 KiB); this prevents
declaring JSON the universal winner. Protobuf adds a schema and binding/toolchain
dependency; indexed binary adds an explicitly versioned fixed record layout and
offset validation plus the same Protobuf event dependency. Its manifest is readable,
but step data need the supplied inspect tool, as does compressed JSON at rest.

Corpus: one local 4096-step stage-11 policy preview, one two-step latest-run replay,
and three seeded synthetic 4096-step fixtures. Retention uses 32 independently
decoded copies of each case, not 32 distinct episodes or a lazy multi-file browser.
The real selected scene regions are mostly empty; they validate world references
and actual rendering helpers, but their CPU timings cannot estimate dense-wall,
GPU or full interactive frame cost. First-frame CPU initialization is present in
raw samples and diluted by pooled medians. No cold-disk, fsync, live append/crash
recovery, dense-world stress or cross-run analytics claims are made.

Before a production decision: collect a diverse representative episode corpus,
test full UI/GPU/async-LOD responsiveness, specify crash-safe writes and schema
evolution, and compare a direct typed reader only if startup or retention is
actually limiting. Benchmark completion does not authorize those migrations.

## Reproducibility and validation

- Base viewer: develop `d65cbfc`; source SHA256
  `b27f89610a35690292269485e2cd6d2eaa61be34f9bc4d6a36c71fea73227f2b`.
- Python 3.12.10, Protobuf 6.33.6, Zstandard 0.25.0; level 3.
  Rust dependencies are frozen by the standalone crate's Cargo.lock.
- All 90 native workers check exact Python-to-Rust JSON-value equality; all five
  cases and six formats completed. Input/source/probe hashes stayed unchanged.
- 23 Python tests and 22 release Rust tests pass. The Rust suite includes 20
  existing viewer tests plus two codec tests. No production renderer edits.
- r3 raw artifacts stay ignored under
  `runtime/trajectory-storage-456-r3/`; commands and inspect/export are in README.
  Preliminary r2 evidence remains separately in RESULTS.md. The discarded r1
  exploratory run is not used.

