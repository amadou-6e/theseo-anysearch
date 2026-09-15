"""Print a compact Markdown report without publishing trajectory payloads."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    lines = ["# Preliminary trajectory storage benchmark (#456)", "",
             "Python codec measurements only; **not Rust/egui viewer timings or a format decision**.",
             "", f"Raw report SHA256: `{hashlib.sha256(raw).hexdigest()}`.", "",
             f"Environment: {report['environment']['platform']}; Python {report['environment']['python'].split()[0]}; "
             f"Protobuf {report['environment']['protobuf']}; Zstandard {report['environment']['zstandard']}.",
             f"Parameters: {report['parameters']['repetitions']} write repetitions, "
             f"{report['parameters']['retained_episodes']} retained independent episode decodes, Zstd level 3.", "",
             "Reads are OS-cached; writes buffered without fsync. Each format/representation uses one fresh "
             "worker process. Load timings need repeated-worker measurements for robust ranking.", ""]
    for case in report["cases"]:
        lines += [f"## {case['name']} ({case['steps']} steps)", "",
                  "| Format | KiB | Write median ms | Native load ms | Row-materialized load ms | Native retained RSS MiB | Rows retained RSS MiB | Native final overlay ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for fmt in case["formats"]:
            native, rows = fmt["loads"]
            def rss(item):
                value = item["after_load_memory"]["rss_bytes"]
                return f"{value / 1048576:.1f}" if value is not None else "unavailable"
            lines.append(f"| {fmt['format']} | {fmt['bytes']/1024:.2f} | {fmt['write']['median_ms']:.2f} | "
                         f"{native['first_load_ms']:.2f} | {rows['first_load_ms'] + rows['first_materialize_ms']:.2f} | "
                         f"{rss(native)} | {rss(rows)} | {native['cumulative_overlay'][-1]['ms']:.2f} |")
        lines += [""]
    lines += ["## Provenance", ""]
    for name, info in report["sources"].items():
        lines.append(f"- {name}: input SHA256 `{info['sha256']}` (local preview/saved trace; no new training).")
    for name, digest in report["source_sha256"].items():
        lines.append(f"- {name}: source SHA256 `{digest}`.")
    lines += ["", "## Disposition and remaining work", "",
              "Retain as preliminary evidence. No production migration or winner selected. "
              "Use the full ignored report for step-access distributions, baseline/peak RSS, "
              "all overlay positions, and artifact hashes. Raw traces/world packs are not committed.", "",
              "Next: shared-schema Rust readers; actual viewer startup and step/overlay/mesh/frame "
              "latency; diverse many-episode corpus; repeated workers; safe format validation and "
              "inspect/export tooling. Fixed binary here uses typed Protobuf event sidecars, not a "
              "dependency-free bespoke mutation format."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print(summarize(parser.parse_args().report), end="")
