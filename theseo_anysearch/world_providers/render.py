"""Deterministic headless PNG previews of complete voxel worlds and routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from theseo_anysearch.world_providers.bundle import VerifiedBundle, sha256

_AXES = ((0, "yz"), (1, "xz"), (2, "xy"))


def render_previews(bundle: VerifiedBundle) -> dict[str, str]:
    """Write orthogonal occupancy projections and start-centered slices."""

    output = bundle.root / "previews"
    output.mkdir(exist_ok=False)
    routes = []
    for reference in bundle.references:
        assert reference.route_artifact is not None
        routes.append(json.loads((bundle.root / reference.route_artifact.relative_path).read_text()))
    font = ImageFont.load_default()
    hashes: dict[str, str] = {}
    for removed_axis, label in _AXES:
        kept = tuple(axis for axis in range(3) if axis != removed_axis)
        slice_index = routes[0][0][removed_axis]
        for mode in ("projection", "slice"):
            if mode == "projection":
                plane = np.max(bundle.occupancy, axis=removed_axis)
            else:
                plane = np.take(bundle.occupancy, slice_index, axis=removed_axis)
            width, height = plane.shape
            scale = max(2, min(6, 600 // max(width, height)))
            image = Image.new("RGB", (width * scale + 24, height * scale + 48), "white")
            draw = ImageDraw.Draw(image)
            for u, v in np.argwhere(plane):
                draw.rectangle(
                    (12 + int(u) * scale, 24 + int(v) * scale,
                     12 + (int(u) + 1) * scale - 1, 24 + (int(v) + 1) * scale - 1),
                    fill=(54, 63, 70),
                )
            for route in routes:
                visible = route if mode == "projection" else [
                    cell for cell in route if cell[removed_axis] == slice_index
                ]
                points = [
                    (12 + (cell[kept[0]] + 0.5) * scale,
                     24 + (cell[kept[1]] + 0.5) * scale)
                    for cell in visible
                ]
                if len(points) > 1 and mode == "projection":
                    draw.line(points, fill=(218, 92, 34), width=max(1, scale // 2))
                elif mode == "slice":
                    for point in points:
                        draw.point(point, fill=(218, 92, 34))
                for cell, color in ((route[0], (20, 130, 84)), (route[-1], (176, 43, 53))):
                    if mode == "slice" and cell[removed_axis] != slice_index:
                        continue
                    point = (12 + (cell[kept[0]] + 0.5) * scale,
                             24 + (cell[kept[1]] + 0.5) * scale)
                    r = max(2, scale // 2)
                    draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=color)
            detail = f"{label} {mode}" + (f" @{slice_index}" if mode == "slice" else "")
            draw.text((12, 5), f"{detail} | {bundle.world.frame.meters_per_voxel:g} m/voxel", fill="black", font=font)
            draw.text((12, height * scale + 29), "dark: solid  orange: route  green: start  red: goal", fill="black", font=font)
            path = output / f"{label}-{mode}.png"
            image.save(path, format="PNG", optimize=False)
            hashes[f"previews/{path.name}"] = sha256(path)
    return hashes
