# Native scenario v2 world queries

This independently compiled extension demonstrates the safe SDK wrapper for point,
bounded-region, bounded-count, and ray queries. Build it with `anysearch compile` or
`cargo build --release` in `extension`. The callback table and all returned values are
valid only while the scenario function is running; the wrapper does not expose raw
pointers to scenario authors.
