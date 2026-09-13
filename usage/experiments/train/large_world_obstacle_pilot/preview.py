"""Generate lightweight replayer entries for the compiled obstacle-world pack.

No training trajectory or voxel enumeration is needed: each entry points to
the immutable pack and places the camera at one preflight route start.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_preview_files(report: dict, output_dir: Path) -> list[Path]:
    """Write one geometry-only replayer entry per sampled world region."""
    pack_path = Path(report["pack_path"]).resolve()
    manifest_path = pack_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["identity_sha256"] != report["world_identity"]:
        raise ValueError("preflight report and compiled-world manifest have different identities")
    extent = list(report["extent"])
    if [manifest["extent"][axis] for axis in ("x", "y", "z")] != extent:
        raise ValueError("preflight report and compiled-world manifest have different extents")

    output_dir.mkdir(parents=True, exist_ok=True)
    world = {
        "identity_sha256": report["world_identity"],
        "schema_version": manifest["schema_version"],
        "coordinate_type": manifest["coordinate_type"],
        "extent": extent,
        "manifest_path": os.path.relpath(manifest_path, output_dir).replace("\\", "/"),
    }
    entries = [
        (f"region_{region['index']:02d}.json", region["start"])
        for region in report["route_regions"]
    ]
    for index, wall in enumerate(report.get("portal_walls", [])):
        side = wall["side"]
        center_y, center_z = wall["center"]
        entries.append((
            f"portal_{index:02d}.json",
            (wall["x"] - 4, center_y - side // 2, center_z),
        ))
    paths = []
    for iteration, (filename, start) in enumerate(entries):
        data = {
            "schema_version": 2,
            "experiment_name": "large-world-obstacle-preview",
            "run_id": report["world_identity"][:12],
            "iteration": iteration,
            "episode_reward_mean": 0.0,
            "grid_size": max(extent),
            "agent_count": 1,
            "max_steps": 0,
            "obs_mode": "box",
            "episode": {
                "total_reward": 0.0,
                "steps_taken": 0,
                "success": False,
                "start_pos": list(start),
                "goal_pos": None,
                "steps": [],
            },
            "world": world,
        }
        path = output_dir / filename
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def write_preview_images(output_dir: Path) -> list[Path]:
    """Render global placement and a local obstacle close-up from source boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from usage.experiments.train.large_world_obstacle_pilot.preflight import (
        EXTENT,
        GLOBAL_SOURCES,
        LOCAL_SOURCES,
        LOCAL_START,
        OFFSETS,
        PORTAL_WALLS,
        REGION_INDICES,
        route_start,
    )

    global_path = output_dir / "global_obstacles.png"
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, sharey=True)
    for ax, layer in zip(axes, (96, 352)):
        for source in GLOBAL_SOURCES:
            x0, y0, _ = source.minimum
            x1, y1, _ = source.maximum_inclusive
            ax.add_patch(Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                                   facecolor="#94a3b8", edgecolor="#64748b", alpha=0.3))
        for x, _ in PORTAL_WALLS:
            ax.axvline(x, color="#1d4ed8", linewidth=1, alpha=0.6)
        for index, (x, y, z) in enumerate(OFFSETS):
            if z != layer:
                continue
            ax.add_patch(Rectangle((x, y), 112, 96, fill=False,
                                   edgecolor="#2563eb", linewidth=1.5))
            ax.text(x + 56, y + 48, str(index), ha="center", va="center", fontsize=8)
        for index in REGION_INDICES:
            if OFFSETS[index][2] == layer:
                x, y, _ = route_start(index)
                ax.scatter(x, y, color="#dc2626", s=48, zorder=5)
        ax.set(xlim=(0, EXTENT[0]), ylim=(0, EXTENT[1]), ylabel="Y (voxels)",
               title=f"Obstacle regions at Z offset {layer}; red dots are preview starts")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("X (voxels)")
    fig.suptitle("4096 x 2048 x 512 compiled world; blue lines are portal walls")
    fig.tight_layout()
    fig.savefig(global_path, dpi=160)
    plt.close(fig)

    local_path = output_dir / "local_obstacles.png"
    fig = plt.figure(figsize=(16, 6))
    ax_xy = fig.add_subplot(1, 3, 1)
    ax_xz = fig.add_subplot(1, 3, 2)
    ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
    for index, source in enumerate(LOCAL_SOURCES):
        x0, y0, z0 = source.minimum
        x1, y1, z1 = source.maximum_inclusive
        color = "#2563eb" if index < 12 else "#f97316"
        if z0 <= LOCAL_START[2] <= z1:
            ax_xy.add_patch(Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                                      facecolor=color, edgecolor="#1e293b"))
        if y0 <= LOCAL_START[1] <= y1:
            ax_xz.add_patch(Rectangle((x0, z0), x1 - x0 + 1, z1 - z0 + 1,
                                      facecolor=color, edgecolor="#1e293b"))
        vertices = (
            (x0, y0, z0), (x1 + 1, y0, z0), (x1 + 1, y1 + 1, z0), (x0, y1 + 1, z0),
            (x0, y0, z1 + 1), (x1 + 1, y0, z1 + 1),
            (x1 + 1, y1 + 1, z1 + 1), (x0, y1 + 1, z1 + 1),
        )
        faces = ([vertices[i] for i in face] for face in (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        ))
        ax_3d.add_collection3d(Poly3DCollection(
            list(faces), facecolors=color, edgecolors="#1e293b",
            linewidths=0.25, alpha=0.25 if index < 12 else 0.65,
        ))
    ax_xy.scatter(LOCAL_START[0], LOCAL_START[1], color="#dc2626", s=35, zorder=5)
    ax_xy.set(xlim=(0, 128), ylim=(0, 96), xlabel="X", ylabel="Y", title="XY slice at Z=32")
    ax_xz.scatter(LOCAL_START[0], LOCAL_START[2], color="#dc2626", s=35, zorder=5)
    ax_xz.set(xlim=(0, 128), ylim=(0, 64), xlabel="X", ylabel="Z", title="XZ slice at Y=48")
    ax_3d.scatter(*LOCAL_START, color="#dc2626", s=25)
    ax_3d.set(xlim=(0, 128), ylim=(0, 96), zlim=(0, 64), xlabel="X", ylabel="Y",
              zlabel="Z", title="One repeated obstacle region")
    ax_3d.set_box_aspect((128, 96, 64))
    ax_3d.view_init(elev=25, azim=-65)
    fig.tight_layout()
    fig.savefig(local_path, dpi=160)
    plt.close(fig)

    portal_path = output_dir / "portal_progression.png"
    fig, axes = plt.subplots(1, len(PORTAL_WALLS), figsize=(16, 4), sharex=True, sharey=True)
    for ax, (x, side) in zip(axes, PORTAL_WALLS):
        ax.add_patch(Rectangle((-20, -20), 40, 40, facecolor="#2563eb"))
        ax.add_patch(Rectangle((-side / 2, -side / 2), side, side,
                               facecolor="white", edgecolor="#1e293b", linewidth=1))
        ax.set(xlim=(-20, 20), ylim=(-20, 20), title=f"X={x}\n{side} x {side}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Central 40 x 40 YZ cutaway of each full-section wall; white is open")
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(portal_path, dpi=160)
    plt.close(fig)
    return [global_path, local_path, portal_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runtime/obstacle-waypoint-pilot/preflight.json"),
    )
    parser.add_argument("--images", action="store_true", help="Also render static PNG views.")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    for path in write_preview_files(report, args.report.parent / "previews"):
        print(path)
    if args.images:
        for path in write_preview_images(args.report.parent / "previews"):
            print(path)


if __name__ == "__main__":
    main()
