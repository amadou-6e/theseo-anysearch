"""Run with python -m usage.benchmarks.trajectory_storage.benchmark --help."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .codecs import FORMATS, json_bytes, read, write


def memory():
    """OS resident/peak resident bytes; no Python-only allocation proxy."""
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": counters.WorkingSetSize, "peak_rss_bytes": counters.PeakWorkingSetSize}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    # ru_maxrss is not current RSS. Report unavailable, rather than mislabel it.
    return {"rss_bytes": None, "peak_rss_bytes": peak}


def timed(call):
    start = time.perf_counter_ns()
    value = call()
    return value, (time.perf_counter_ns() - start) / 1e6


def distribution(values):
    ordered = sorted(values)
    return {"median_ms": statistics.median(values),
            "p95_ms": ordered[min(len(values) - 1, int(len(values) * .95))]}


def worker(directory, kind, count, representation):
    baseline = memory()
    loaded, first_ms = timed(lambda: read(directory, kind))
    materialize_ms = 0.0
    if representation == "rows":
        rows, materialize_ms = timed(loaded.export)
        from .codecs import Loaded, split
        meta, steps = split(rows)
        loaded = Loaded(meta, steps, "json")
        del rows
    retained = [loaded]
    load_times = []
    for _ in range(count - 1):
        def load_one():
            value = read(directory, kind)
            if representation == "rows":
                from .codecs import Loaded, split
                meta, steps = split(value.export())
                return Loaded(meta, steps, "json")
            return value
        value, elapsed = timed(load_one)
        retained.append(value)
        load_times.append(elapsed)
    after_load = memory()
    randomizer = random.Random(456)
    queries = [randomizer.randrange(len(loaded)) for _ in range(128)]
    access = [timed(lambda: loaded.step(index))[1] for index in queries]
    overlay = []
    for index in sorted({0, len(loaded) // 4, len(loaded) // 2, len(loaded) - 1}):
        state, elapsed = timed(lambda: loaded.overlay(index))
        overlay.append({"step": index, "ms": elapsed, "modified_coordinates": len(state)})
    return {"representation": representation, "retained_episodes": len(retained),
            "first_load_ms": first_ms, "first_materialize_ms": materialize_ms,
            "subsequent_load": distribution(load_times) if load_times else None,
            "baseline_memory": baseline, "after_load_memory": after_load,
            "step_access": distribution(access), "cumulative_overlay": overlay}


def synthetic(count=4096, agents=1, mutation_every=0):
    rng = random.Random(456 + agents + mutation_every)
    steps = []
    for i in range(count):
        value = {"step": i, "action": rng.randrange(18), "reward": rng.random() - .5,
                 "done": i == count - 1, "cursor_x": 70_000 + i,
                 "cursor_y": 1024 + i % 113, "cursor_z": 256,
                 "voxel_count": i // max(mutation_every, 1), "placed": False, "mutations": []}
        if mutation_every and i % mutation_every == 0:
            value["mutations"] = [{"coordinate": [70_000 + i % 64, 1024, 256],
                "occupied": bool((i // mutation_every) % 2), "kind": 2,
                "active": True, "reward_weight": -.123456789012345}]
        if agents > 1:
            value.update(actions=[rng.randrange(18) for _ in range(agents)],
                         rewards=[rng.random() for _ in range(agents)],
                         cursors=[[70_000 + i, 1000 + a, 256] for a in range(agents)],
                         placed_per_agent=[False] * agents, dones=[i == count - 1] * agents)
        steps.append(value)
    return {"schema_version": 2, "experiment_name": "synthetic-storage-benchmark",
            "run_id": "not-a-training-run", "agent_count": agents,
            "world": {"identity_sha256": "synthetic-no-world-pack"},
            "episode": {"steps": steps, "steps_taken": count, "success": True}}


def run(args):
    import google.protobuf
    import zstandard
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(__file__).with_name("codecs.py"),
                    Path(__file__).with_name("trajectory_bench.proto")]
    source_hashes_at_start = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    cases = [("synthetic-single", synthetic()),
             ("synthetic-mutations", synthetic(mutation_every=16)),
             ("synthetic-multi", synthetic(agents=4, mutation_every=16))]
    source_hashes = {}
    for i, path in enumerate(args.inputs):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not payload.get("episode", {}).get("steps"):
            raise ValueError(f"not a nonempty trajectory: {path}")
        name = f"real-{i}-{path.stem}"
        cases.append((name, payload))
        source_hashes[name] = {"source": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    report = {"benchmark_version": 1, "source_sha256": source_hashes_at_start, "environment": {
        "python": sys.version, "platform": platform.platform(),
        "protobuf": google.protobuf.__version__, "zstandard": zstandard.__version__,
        "cpu_count": os.cpu_count()}, "parameters": {
        "repetitions": args.repetitions, "retained_episodes": args.episodes,
        "zstd_level": 3, "seed": 456}, "limitations": [
            "OS-cached reads; no cache eviction or cold-disk claim.",
            "Buffered file writes, not fsync durability timing.",
            "Python codec/access/overlay timings, NOT actual Rust/egui startup or render latency.",
            "Repeated same episode models retention; not a diverse run corpus.",
            "World packs are referenced and excluded from size; fixed binary uses typed protobuf event sidecar.",
            "Two representations: native codec data vs Python dictionaries; neither measures Rust allocation size.",
        ], "sources": source_hashes, "cases": []}
    for name, payload in cases:
        case = {"name": name, "steps": len(payload["episode"]["steps"]),
                "payload_sha256": hashlib.sha256(json_bytes(payload)).hexdigest(), "formats": []}
        for kind in args.formats:
            directory = args.output / name / kind
            writes = [timed(lambda: write(directory, kind, payload))[1] for _ in range(args.repetitions)]
            restored = read(directory, kind).export()
            if restored != payload or json_bytes(restored) != json_bytes(payload):
                # Dict ordering differs between codecs; compare canonical JSON instead.
                if json.dumps(restored, sort_keys=True) != json.dumps(payload, sort_keys=True):
                    raise AssertionError(f"round-trip mismatch: {name}/{kind}")
            result = {"format": kind, "bytes": sum(p.stat().st_size for p in directory.iterdir()),
                      "write": distribution(writes), "round_trip": True,
                      "artifact_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in directory.iterdir()}, "loads": []}
            for representation in ("native", "rows"):
                command = [sys.executable, "-m", "usage.benchmarks.trajectory_storage.benchmark",
                           "--worker", str(directory.resolve()), "--format", kind,
                           "--episodes", str(args.episodes), "--representation", representation]
                start = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, text=True, check=True)
                measurement = json.loads(completed.stdout)
                measurement["worker_wall_ms"] = (time.perf_counter() - start) * 1000
                result["loads"].append(measurement)
            case["formats"].append(result)
            print(f"{name}: {kind}, {result['bytes']} bytes, write {result['write']['median_ms']:.2f} ms", flush=True)
        report["cases"].append(case)
        (args.output / "report.json").write_bytes(json_bytes(report, pretty=True))
    if source_hashes_at_start != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}:
        raise RuntimeError("benchmark source changed during execution; discard this run")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runtime/trajectory-storage-456"))
    parser.add_argument("--inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--format", choices=FORMATS)
    parser.add_argument("--representation", choices=("native", "rows"), default="native")
    args = parser.parse_args()
    if args.episodes < 1 or args.repetitions < 1:
        parser.error("episodes and repetitions must be positive")
    if args.worker:
        if not args.format:
            parser.error("worker requires --format")
        print(json.dumps(worker(args.worker, args.format, args.episodes, args.representation)))
    else:
        run(args)


if __name__ == "__main__":
    main()
