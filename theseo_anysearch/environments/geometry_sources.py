"""Resolve canonical small-world geometry sources by voxel union."""

from __future__ import annotations

from typing import Any, Callable


def resolve_geometry_sources(
    config: dict[str, Any],
    *,
    grid_size: int,
    load_stl: Callable[..., list[tuple[int, int, int]]],
) -> list[tuple[int, int, int]]:
    """Materialize ordered sources with deterministic set-union semantics."""

    occupied: set[tuple[int, int, int]] = set()
    for source in config.get("geometry_sources") or []:
        if source["type"] == "stl":
            occupied.update(
                load_stl(
                    str(source["path"]),
                    float(source.get("scale", 1.0)),
                    grid_size,
                    padding=int(source.get("padding", 2)),
                )
            )
        elif source["type"] == "boxes":
            for xmin, ymin, zmin, xmax, ymax, zmax in source["boxes"]:
                occupied.update(
                    (x, y, z)
                    for x in range(xmin, xmax + 1)
                    for y in range(ymin, ymax + 1)
                    for z in range(zmin, zmax + 1)
                )
        else:
            raise ValueError(f"unsupported geometry source type: {source['type']}")
    return sorted(occupied)
