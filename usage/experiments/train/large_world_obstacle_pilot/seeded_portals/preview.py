"""Plot actual A* wall crossings for two seed-selected gate worlds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theseo_anysearch.worlds.seeded_catalog import load_catalog
from usage.experiments.train.large_world_obstacle_pilot.preflight import PORTAL_WALLS


def write_preview(catalog_path: Path, preflight_path: Path, output: Path) -> Path:
    catalog = load_catalog(catalog_path)
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if report["catalog_sha256"] != catalog.identity_sha256:
        raise ValueError("preview report and world catalog do not match")
    selected = [item for item in report["results"] if item["stage"] == 11][:2]
    if len(selected) != 2:
        raise ValueError("preview needs two complete final-stage trajectories")
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
    for axis, item in zip(axes, selected):
        variant = catalog.for_seed(item["seed"])
        axis.set_facecolor("#101820")
        for (x, side), center in zip(PORTAL_WALLS, variant.portal_centers):
            axis.axvline(x, color="#87909b", linewidth=1.5, alpha=0.8)
            axis.scatter([x], [center[0]], color="#55df91", marker="s",
                         s=max(60, side * 8), edgecolor="#d9ffe9", zorder=4)
            axis.annotate(f"{side}×{side}", (x, center[0]),
                          xytext=(3, 8), textcoords="offset points",
                          color="white", fontsize=8)
        points = [(1, 1024), *[(x, y) for x, y, _ in item["wall_crossings"]],
                  (4095, variant.portal_centers[-1][0])]
        axis.plot(*zip(*points), color="#ffd54d", marker="o", linewidth=2.2,
                  markersize=3, label="A* wall crossings")
        axis.set_ylim(820, 1220)
        axis.set_ylabel("Y along wall (voxels)", color="#202a33")
        axis.set_title(
            f"Episode seed {item['seed']} → world {variant.identity_sha256[:12]}  "
            f"|  {item['lateral_actions']} lateral actions / 4096",
            color="#202a33", fontsize=11,
        )
        axis.tick_params(colors="#202a33")
        axis.grid(color="#405060", alpha=0.5)
    axes[-1].set_xlim(0, 4096)
    axes[-1].set_xlabel("X through six full-section walls (voxels)", color="#202a33")
    fig.suptitle("Seeded staggered portals: only the green square is passable at each wall",
                 fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("preflight", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(write_preview(args.catalog, args.preflight, args.output))


if __name__ == "__main__":
    main()
