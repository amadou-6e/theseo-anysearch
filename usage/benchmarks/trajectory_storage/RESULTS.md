# Preliminary trajectory storage benchmark (#456)

Python codec measurements only; **not Rust/egui viewer timings or a format decision**.

Raw report SHA256: `a9a506dd8793ad46f2eb059be047ad9432243d6bacc307700739637b0c76cf1b`.

Environment: Windows-10-10.0.19045-SP0; Python 3.12.10; Protobuf 6.33.6; Zstandard 0.25.0.
Parameters: 3 write repetitions, 32 retained independent episode decodes, Zstd level 3.

Reads are OS-cached; writes buffered without fsync. Each format/representation uses one fresh worker process. Load timings need repeated-worker measurements for robust ranking.

## synthetic-single (4096 steps)

| Format | KiB | Write median ms | Native load ms | Row-materialized load ms | Native retained RSS MiB | Rows retained RSS MiB | Native final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1112.54 | 41.10 | 9.78 | 10.27 | 84.8 | 85.1 | 1.29 |
| json-zstd | 96.39 | 10.03 | 11.41 | 12.04 | 85.2 | 85.0 | 1.27 |
| protobuf | 127.59 | 56.99 | 0.68 | 21.07 | 39.3 | 85.0 | 18.81 |
| protobuf-zstd | 71.01 | 56.95 | 2.68 | 23.70 | 39.7 | 85.4 | 19.21 |
| binary | 220.41 | 49.20 | 1.20 | 27.60 | 32.8 | 84.9 | 23.82 |
| binary-zstd | 80.12 | 49.87 | 3.18 | 29.10 | 33.4 | 86.4 | 24.26 |

## synthetic-mutations (4096 steps)

| Format | KiB | Write median ms | Native load ms | Row-materialized load ms | Native retained RSS MiB | Rows retained RSS MiB | Native final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1173.97 | 39.95 | 9.85 | 10.90 | 84.4 | 84.5 | 1.35 |
| json-zstd | 97.31 | 10.49 | 11.19 | 12.55 | 84.6 | 85.1 | 1.33 |
| protobuf | 132.66 | 57.12 | 0.68 | 22.17 | 40.7 | 84.4 | 20.05 |
| protobuf-zstd | 71.74 | 58.01 | 2.50 | 24.06 | 41.0 | 85.0 | 20.15 |
| binary | 227.41 | 50.04 | 1.16 | 27.30 | 33.2 | 84.6 | 24.65 |
| binary-zstd | 74.95 | 52.04 | 3.13 | 30.97 | 33.8 | 85.3 | 24.47 |

## synthetic-multi (4096 steps)

| Format | KiB | Write median ms | Native load ms | Row-materialized load ms | Native retained RSS MiB | Rows retained RSS MiB | Native final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 4313.38 | 152.92 | 30.32 | 31.27 | 265.9 | 265.9 | 1.48 |
| json-zstd | 323.86 | 33.08 | 44.31 | 32.91 | 266.7 | 266.4 | 1.51 |
| protobuf | 556.66 | 131.31 | 2.56 | 68.45 | 95.4 | 265.9 | 58.92 |
| protobuf-zstd | 262.92 | 131.80 | 4.87 | 69.97 | 95.7 | 265.9 | 61.98 |
| binary | 647.41 | 121.65 | 1.70 | 79.06 | 46.1 | 266.1 | 66.83 |
| binary-zstd | 257.51 | 127.11 | 4.33 | 73.21 | 47.2 | 266.5 | 63.16 |

## real-0-iter_000300 (4096 steps)

| Format | KiB | Write median ms | Native load ms | Row-materialized load ms | Native retained RSS MiB | Rows retained RSS MiB | Native final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1034.91 | 36.76 | 9.99 | 9.54 | 81.0 | 81.4 | 1.91 |
| json-zstd | 8.29 | 6.50 | 13.04 | 11.01 | 81.5 | 81.7 | 1.30 |
| protobuf | 76.46 | 58.08 | 0.57 | 26.04 | 39.4 | 80.8 | 19.48 |
| protobuf-zstd | 9.71 | 57.35 | 2.98 | 23.03 | 39.8 | 81.4 | 19.51 |
| binary | 221.04 | 53.92 | 1.19 | 27.08 | 32.9 | 81.6 | 24.10 |
| binary-zstd | 16.18 | 52.30 | 3.12 | 29.98 | 33.5 | 82.5 | 30.93 |

## real-1-iter_000100 (2 steps)

| Format | KiB | Write median ms | Native load ms | Row-materialized load ms | Native retained RSS MiB | Rows retained RSS MiB | Native final overlay ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| json-pretty | 1.31 | 0.64 | 0.23 | 0.22 | 21.7 | 21.8 | 0.00 |
| json-zstd | 0.44 | 0.87 | 1.94 | 1.95 | 22.1 | 21.9 | 0.00 |
| protobuf | 0.63 | 0.85 | 0.22 | 0.23 | 21.8 | 21.8 | 0.01 |
| protobuf-zstd | 0.42 | 0.68 | 2.16 | 2.00 | 21.9 | 22.0 | 0.01 |
| binary | 1.11 | 2.03 | 0.58 | 0.68 | 21.8 | 21.9 | 0.01 |
| binary-zstd | 1.09 | 2.30 | 4.19 | 2.54 | 22.3 | 22.5 | 0.02 |

## Provenance

- real-0-iter_000300: input SHA256 `fda5337188b31ad10a50f0218b7c597dd555710f3b65d213ee9288d02dfb221a` (local preview/saved trace; no new training).
- real-1-iter_000100: input SHA256 `36c3719c8425c7e60eb74c9d79ac330cc4aa11ff7988c8303d58cb51ea8b46a7` (local preview/saved trace; no new training).
- benchmark.py: source SHA256 `4b06166dabca34de0eb082e42d348dec0d6e2e1df1a4cc6e66e681e59f93b038`.
- codecs.py: source SHA256 `a0f80d6c2e7cf5bdd91312745cf7766d0679ddfc7c8f4a76b80f385f57da49e8`.
- trajectory_bench.proto: source SHA256 `baa21ffde908c17806fb68d53a8342aa64d8467e38ace2ae4b4a23e64303552d`.

## Disposition and remaining work

Retain as preliminary evidence. No production migration or winner selected. Use the full ignored report for step-access distributions, baseline/peak RSS, all overlay positions, and artifact hashes. Raw traces/world packs are not committed.

Next: shared-schema Rust readers; actual viewer startup and step/overlay/mesh/frame latency; diverse many-episode corpus; repeated workers; safe format validation and inspect/export tooling. Fixed binary here uses typed Protobuf event sidecars, not a dependency-free bespoke mutation format.
