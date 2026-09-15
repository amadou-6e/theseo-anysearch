# Regional replay benchmark

Run from `theseo_anysearch/core` with:

```text
cargo run --release --bin regional-replay-bench
```

The fixture is a 60,000 x 40,000 x 20,000 compiled world containing 16 sparse
chunks. It performs one cold radius-16 frame followed by 100 frame-preparation
samples across the sparse locations. Results recorded on Windows x86-64 on
31 August 2026:

| Measurement | Result |
| --- | ---: |
| First visible frame | 0.2990 ms |
| Frame preparation p50 | 0.0484 ms |
| Frame preparation p95 | 0.1782 ms |
| Viewer process RSS | 10,637,312 bytes |
| Visible voxels in final frame | 1 |
| Resident chunks | 16 |
| Pack reads | 16 |
| Cache hits / misses | 186 / 16 |

The pack performed one cold read per sparse chunk and no reads proportional to
the logical world extent. RSS remained approximately 10.6 MB, demonstrating
that opening the large manifest and navigating between its sparse regions does
not materialize the full world.
