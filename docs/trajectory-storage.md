# Trajectory storage

New single-agent, multi-agent and heuristic snapshots use compact UTF-8 JSON
compressed with Zstandard level 3, named `iter_000001.json.zst`, `best.json.zst`
or `heuristic_astar.json.zst`. The existing trajectory JSON schema is unchanged.
Metadata (`best_meta.json`), immutable world packs and `.npy` geometry sidecars
remain separate and unchanged. References resolve relative to the saved file.

Python and the native replayer load both legacy `.json` and new `.json.zst`.
Decompression happens once at load; steps remain indexed in memory for scrubbing.
Discovery prefers compressed files when both encodings exist for an iteration,
but a corrupt compressed file is an error, not a reason to silently replay stale
legacy data. Existing artifacts are not rewritten or deleted.

Writers serialize a complete finite-valued JSON document, compress it, write and
fsync a temporary sibling, then atomically replace the destination. Failed writes
leave the old snapshot intact. This is complete-snapshot publication, not live
step append; directory durability after power loss depends on the filesystem.

For human-readable inspection or export:

```powershell
python -m theseo_anysearch.experiments.trajectory_storage path/to/best.json.zst
python -m theseo_anysearch.experiments.trajectory_storage path/to/best.json.zst --step 100
python -m theseo_anysearch.experiments.trajectory_storage path/to/best.json.zst --output readable.json
```

Export refuses to overwrite an existing file. Replaying by path and selecting
best/latest/iterations work for either encoding; metadata is not a replay file.

The choice follows #456: compressed JSON was smaller and cheaper to write on the
real long trace, while Rust load differences were modest and all formats scrubbed
the same materialized step vector. It avoids a custom binary schema and generated
bindings. Other workloads may favor another encoding; this does not claim GPU,
dense-world or cross-run analytics performance improvements.
