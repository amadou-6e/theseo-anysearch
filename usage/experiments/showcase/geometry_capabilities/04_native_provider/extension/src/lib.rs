use anysearch_extension::{anysearch_geometry_v1, GeometryContextV1};

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 {
    anysearch_extension::ABI_VERSION
}
#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    64 // CAP_GEOMETRY
}

/// Mirrors the Python `wall` provider in the sibling `geometry.py` exactly:
/// a single wall at `wall_x` with a one-voxel gap at `gap_z`, driven only by
/// `context.parameters`/`context.seed` -- no randomness, so both providers
/// produce byte-comparable proposals from the same inputs.
#[anysearch_geometry_v1]
fn wall(context: &GeometryContextV1<'_>) -> serde_json::Value {
    let x = context
        .parameters
        .get("wall_x")
        .and_then(|value| value.as_i64())
        .unwrap_or(16);
    let gap_z = context
        .parameters
        .get("gap_z")
        .and_then(|value| value.as_i64())
        .unwrap_or(8);

    let mut boxes = Vec::new();
    if gap_z - 1 >= 1 {
        boxes.push(serde_json::json!([x, 1, 1, x, 30, gap_z - 1]));
    }
    if gap_z + 1 <= 30 {
        boxes.push(serde_json::json!([x, 1, gap_z + 1, x, 30, 30]));
    }

    serde_json::json!({
        "proposal_id": format!("wall-{}-{}-{}", context.seed, x, gap_z),
        "version": "1",
        "sources": [{"type": "boxes", "boxes": boxes}],
        "metadata": {"wall_x": x, "gap_z": gap_z}
    })
}
