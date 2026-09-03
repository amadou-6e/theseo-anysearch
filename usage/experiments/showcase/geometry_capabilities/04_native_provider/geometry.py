"""Python reference provider mirrored by extension/src/lib.rs's Rust export.

Builds a single wall at ``wall_x`` with a one-voxel gap at ``gap_z``, purely
from ``context.parameters``/``context.seed`` -- no randomness, so the Rust
export below can reproduce byte-comparable output from the same inputs.
"""


def wall(context):
    x = int(context.parameters.get("wall_x", 16))
    gap_z = int(context.parameters.get("gap_z", 8))
    boxes = []
    if gap_z - 1 >= 1:
        boxes.append([x, 1, 1, x, 30, gap_z - 1])
    if gap_z + 1 <= 30:
        boxes.append([x, 1, gap_z + 1, x, 30, 30])
    return {
        "proposal_id": f"wall-{context.seed}-{x}-{gap_z}",
        "version": "1",
        "sources": [{"type": "boxes", "boxes": boxes}],
        "metadata": {"wall_x": x, "gap_z": gap_z},
    }
