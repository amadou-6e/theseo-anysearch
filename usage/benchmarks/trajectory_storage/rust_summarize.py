"""Print a compact Markdown table from paired Python/Rust reports."""
import argparse
import hashlib
import json
import statistics


def render(python_path, rust_path):
    py = json.loads(python_path.read_bytes())
    rust = json.loads(rust_path.read_bytes())
    if hashlib.sha256(python_path.read_bytes()).hexdigest() != rust["python_report_sha256"]:
        raise ValueError("reports are not from the same inputs")
    output = ["# Native replay storage measurements (#456)", "",
              f"Python report SHA256: `{rust['python_report_sha256']}`.", "",
              f"Rust report SHA256: `{hashlib.sha256(rust_path.read_bytes()).hexdigest()}`.", "",
              f"Probe SHA256: `{rust['probe_sha256']}`; {rust['rustc']}.", "",
              f"{rust['platform']}; {rust['repetitions']} fresh release workers per format; "
              f"{rust['retained_episodes']} independently decoded copies retained per worker.", "",
              "Cached reads, buffered writes without fsync. Load includes prototype decoding through serde_json::Value "
              "and conversion to the actual viewer StepData vector. RSS is total working set, not allocation accounting. "
              "Scrub record access and actual cumulative mutation reconstruction are separate from CPU scene work. "
              "GPU presentation and full interactive/async-LOD latency are not measured.", ""]
    for pc, rc in zip(py["cases"], rust["cases"]):
        if pc["name"] != rc["name"]:
            raise ValueError("case mismatch")
        output += [f"## {rc['name']} ({rc['steps']} steps)", "",
                   "| Format | KiB | Write ms | Rust load ms | Retained RSS MiB | Record p95 us | Final overlay ms |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for pf, rf in zip(pc["formats"], rc["formats"]):
            if pf["format"] != rf["format"]:
                raise ValueError("format mismatch")
            samples = rf["samples"]
            median = lambda fn: statistics.median(fn(s) for s in samples)
            output.append(f"| {rf['format']} | {pf['bytes']/1024:.2f} | {pf['write']['median_ms']:.2f} | "
                          f"{rf['median_load_materialize_ms']:.2f} | "
                          f"{median(lambda s: s['after_load_memory']['rss_bytes'])/1024**2:.1f} | "
                          f"{median(lambda s: s['step_access_p95_ms'])*1000:.2f} | "
                          f"{median(lambda s: s['actual_viewer_overlay'][-1]['ms']):.3f} |")
        scenes = [s["scene"] for row in rc["formats"] for s in row["samples"] if "cpu_scene_samples" in s["scene"]]
        if scenes:
            frames = [frame for scene in scenes for frame in scene["cpu_scene_samples"]]
            median = lambda values: statistics.median(values)
            output += ["", "Format-independent actual-world scene probe (pooled across formats/repeats): "
                       f"original-artifact load {median([s['actual_artifact_load_ms'] for s in scenes]):.2f} ms; "
                       f"viewer world setup {median([s['actual_viewer_world_setup_ms'] for s in scenes]):.2f} ms; "
                       f"regional overlay/load/mesh/sort {median([f['region_overlay_mesh_ms'] for f in frames]):.2f} ms; "
                       f"CPU paint/tessellate {median([f['scene_paint_tessellate_ms'] for f in frames]):.2f} ms. "
                       f"Selected regions contain {min(f['faces'] for f in frames)}-{max(f['faces'] for f in frames)} faces. "
                       "Four selected steps, synchronous radius 16, actual scene helpers; not a dense-wall stress test or an end-to-end GUI startup/frame benchmark."]
        output += [""]
    return "\n".join(output)


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_report", type=Path)
    parser.add_argument("rust_report", type=Path)
    args = parser.parse_args()
    print(render(args.python_report, args.rust_report))
