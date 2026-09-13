"""Generate lightweight replayer entries for the compiled obstacle-world pack.

No training trajectory or voxel enumeration is needed: each entry points to
the immutable pack and places the camera just before one portal wall.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_preview_files(report: dict, output_dir: Path) -> list[Path]:
    """Write one geometry-only replayer entry per portal wall."""
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
    entries = []
    for index, wall in enumerate(report.get("portal_walls", [])):
        center_y, center_z = wall["center"]
        entries.append((
            f"portal_{index:02d}.json",
            (wall["x"] - 4, center_y, center_z),
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
    """Render wall locations and equal-scale aperture cutaways."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    from usage.experiments.train.large_world_obstacle_pilot.preflight import (
        EXTENT,
        PORTAL_CENTER,
        PORTAL_WALLS,
    )

    global_path = output_dir / "global_obstacles.png"
    fig, ax = plt.subplots(figsize=(14, 7))
    for x, side in PORTAL_WALLS:
        ax.axvline(x, color="#2563eb", linewidth=1.5)
        ax.scatter(x, PORTAL_CENTER[0], color="#f97316", s=36, zorder=5)
        ax.text(x, PORTAL_CENTER[0] + 60, f"{side} x {side}",
                ha="center", fontsize=9)
    ax.set(xlim=(0, EXTENT[0]), ylim=(0, EXTENT[1]), xlabel="X (voxels)",
           ylabel="Y (voxels)",
           title=f"Full YZ-section walls; portals centered at Y={PORTAL_CENTER[0]}, Z={PORTAL_CENTER[1]}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(global_path, dpi=160)
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
    return [global_path, portal_path]


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
