"""Seeded procedural obstacle provider for the geometry-capabilities showcase.

Places ``wall_count`` full-height walls at random x positions, each pierced by
a single one-voxel-wide gap at a random y. Every wall blocks the full z range,
so the only way through is finding each gap in turn -- a small, deterministic
routing puzzle whose layout is fully determined by ``context.seed``.
"""


def procedural_walls(context):
    import random

    rng = random.Random(context.seed)
    wall_count = int(context.parameters.get("wall_count", 2))
    extent = context.extent
    margin = 4
    candidates = list(range(margin, extent[0] - margin))
    xs = sorted(rng.sample(candidates, min(wall_count, len(candidates))))

    boxes = []
    walls = []
    for x in xs:
        gap_y = rng.randint(margin, extent[1] - margin)
        if gap_y - 1 >= 2:
            boxes.append([x, 2, 2, x, gap_y - 1, extent[2] - 2])
        if gap_y + 1 <= extent[1] - 1:
            boxes.append([x, gap_y + 1, 2, x, extent[1] - 1, extent[2] - 2])
        walls.append({"x": x, "gap_y": gap_y})

    return {
        "proposal_id": f"procedural-walls-{context.seed}-{wall_count}",
        "version": "1",
        "sources": [{"type": "boxes", "boxes": boxes}],
        "metadata": {"walls": walls},
    }
