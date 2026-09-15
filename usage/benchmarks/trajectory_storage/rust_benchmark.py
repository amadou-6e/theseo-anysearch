"""Repeated release-mode Rust reader and actual replay-helper measurements."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from .benchmark import synthetic
from .codecs import json_bytes


def run(args):
    report = json.loads(args.report.read_bytes())
    root = Path(__file__).resolve().parents[3]
    source_paths = [Path(__file__), Path(__file__).parent / "trajectory_bench.proto",
                    *sorted((Path(__file__).parent / "rust").glob("src/*.rs")),
                    Path(__file__).parent / "rust/Cargo.toml",
                    Path(__file__).parent / "rust/Cargo.lock",
                    Path(__file__).parent / "rust/build.rs",
                    root / "theseo_anysearch/core/src/bin/voxel_replay.rs"]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    sources = dict(report["sources"])
    generated = {
        "synthetic-single": synthetic(),
        "synthetic-mutations": synthetic(mutation_every=16),
        "synthetic-multi": synthetic(agents=4, mutation_every=16),
    }
    for name, payload in generated.items():
        path = args.report.parent / (name + "-input.json")
        path.write_bytes(json_bytes(payload))
        sources[name] = {"source": str(path.resolve())}
    result = {"platform": platform.platform(), "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              "python_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "source_sha256": source_hashes,
              "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
              "repetitions": args.repetitions, "retained_episodes": args.episodes,
              "limits": ["OS-cached reads; release profile; repeated fresh processes.",
                         "All codecs materialize into the viewer StepData vector, not zero-copy integration.",
                         "CPU scene probe reuses actual viewer helpers; excludes GPU upload/presentation and full sidebar UI.",
                         "Scene samples are format-independent verified original artifacts, synchronous radius-16 loading, not async LOD scheduling.",
                         "Synthetic inputs have no real world pack; real artifact failures are fatal.",
                         "Repeated same episode approximates retained memory; not a diverse corpus."], "cases": []}
    for case in report["cases"]:
        source = Path(sources[case["name"]]["source"])
        original = json.loads(source.read_bytes())
        if hashlib.sha256(json_bytes(original)).hexdigest() != case["payload_sha256"]:
            raise ValueError(f"input does not match Python report: {case['name']}")
        rows = []
        for fmt in case["formats"]:
            directory = args.report.parent / case["name"] / fmt["format"]
            samples = []
            for _ in range(args.repetitions):
                start = time.perf_counter()
                completed = subprocess.run([str(args.probe.resolve()), str(directory.resolve()),
                    fmt["format"], str(args.episodes), str(source.resolve())],
                    text=True, capture_output=True, check=True)
                probe = json.loads(completed.stdout)
                if probe["build_profile"] != "release":
                    raise ValueError("Rust probe must use release profile")
                if not case["name"].startswith("synthetic") and "unavailable" in probe["scene"]:
                    raise ValueError(f"real artifact scene failed: {probe['scene']}")
                probe["worker_wall_ms"] = (time.perf_counter() - start) * 1000
                samples.append(probe)
            rows.append({"format": fmt["format"], "samples": samples,
                         "median_load_materialize_ms": statistics.median(s["decode_ms"] + s["viewer_materialize_ms"] for s in samples)})
            print(f"{case['name']}: {fmt['format']} Rust load+materialize {rows[-1]['median_load_materialize_ms']:.2f} ms", flush=True)
        result["cases"].append({"name":case["name"], "steps":case["steps"], "formats":rows})
        args.output.write_bytes(json_bytes(result, pretty=True))
    if hashlib.sha256(args.probe.read_bytes()).hexdigest() != result["probe_sha256"]:
        raise RuntimeError("probe changed during measurement; discard run")
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != source_hashes[str(p.relative_to(root))] for p in source_paths):
        raise RuntimeError("probe sources changed during measurement; discard run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1 or args.repetitions < 1:
        parser.error("episodes and repetitions must be positive")
    run(args)
