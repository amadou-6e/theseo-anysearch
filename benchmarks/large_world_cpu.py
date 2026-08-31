"""Single-CPU phase benchmark for sparse compiled finite worlds."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import theseo_core

from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent

EXTENT = (60_000, 40_000, 20_000)
CACHE_BYTES = 64 * 1024 * 1024


class _WindowsMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_memory() -> tuple[int, int]:
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsMemoryCounters),
        ctypes.c_ulong,
    ]
    counters = _WindowsMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError()
    return counters.WorkingSetSize, counters.PeakWorkingSetSize


def _memory_bytes() -> tuple[int, int | None]:
    if os.name == "nt":
        return _windows_memory()
    statm = Path("/proc", "self", "statm")
    if statm.is_file():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE"), None
    return 0, None


def _pin_to_one_cpu() -> int:
    if os.name != "nt" and hasattr(os, "sched_getaffinity"):
        selected = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {selected})
        return selected
    if os.name != "nt":
        raise RuntimeError("single-CPU affinity is unsupported on this platform")
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process = kernel32.GetCurrentProcess()
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ctypes.WinError()
    selected_mask = process_mask.value & -process_mask.value
    if not kernel32.SetProcessAffinityMask(process, selected_mask):
        raise ctypes.WinError()
    return selected_mask.bit_length() - 1


def _points(count: int) -> list[tuple[int, int, int]]:
    return [
        (
            64 + (index * 7_919) % 59_800,
            64 + (index * 5_003) % 39_800,
            64 + (index * 3_007) % 19_800,
        )
        for index in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reset", "scalar", "box", "mask"))
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--box-radius", type=int, default=4)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("runtime", "large-world-cpu")
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.box_radius < 0:
        raise ValueError("box-radius cannot be negative")
    logical_cpu = _pin_to_one_cpu()
    locations = _points(max(args.iterations, 512))
    world = compile_world(
        [BoxSource(minimum=point, maximum_inclusive=point) for point in locations[:512]],
        WorldExtent.from_value(EXTENT),
        args.cache_dir.joinpath("packs"),
    )
    environment = theseo_core.PyVoxelEnv(
        max_steps=args.iterations + 1,
        trail_mode=False,
        extent=EXTENT,
        box_radius=args.box_radius if args.phase == "box" else None,
    )
    environment.set_compiled_world(str(world.root.resolve()), CACHE_BYTES)
    environment.set_world_residency_radius(7)
    start = locations[0]
    environment.set_waypoints(start, (start[0] + 100, start[1], start[2]), 100)
    environment.reset(1)
    initial_rss, initial_peak = _memory_bytes()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for index in range(args.iterations):
        if args.phase == "reset":
            point = locations[index]
            environment.set_waypoints(point, (point[0] + 1, point[1], point[2]), 1)
            environment.reset(index + 2)
        elif args.phase == "mask":
            environment.action_mask()
        else:
            environment.step(26)
    final_rss, final_peak = _memory_bytes()
    print(
        json.dumps(
            {
                "phase": args.phase,
                "box_radius": args.box_radius if args.phase == "box" else None,
                "iterations": args.iterations,
                "logical_cpu": logical_cpu,
                "extent": EXTENT,
                "logical_voxels": EXTENT[0] * EXTENT[1] * EXTENT[2],
                "wall_seconds": time.perf_counter() - wall_started,
                "cpu_seconds": time.process_time() - cpu_started,
                "initial_rss_bytes": initial_rss,
                "final_rss_bytes": final_rss,
                "peak_rss_bytes": final_peak or initial_peak,
                "cache_limit_bytes": CACHE_BYTES,
                "cache_metrics": environment.world_cache_metrics(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
