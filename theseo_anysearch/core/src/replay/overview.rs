//! Bounded compiled-world overview loading and camera-relative projection.

use std::{fs, path::Path};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"AOM1";
const HEADER_BYTES: usize = 16;
const SUPPORTED_ALGORITHM_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct OverviewMesh {
    pub vertices: Vec<[u32; 3]>,
    pub indices: Vec<u32>,
    pub extent: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedVertex {
    pub x: f32,
    pub y: f32,
    pub depth: f32,
}

#[derive(Deserialize)]
struct Manifest {
    identity_sha256: String,
    extent: AxisExtent,
    #[serde(default)]
    overview: Option<OverviewEntry>,
}

#[derive(Deserialize)]
struct AxisExtent {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Deserialize)]
struct OverviewEntry {
    schema_version: u32,
    relative_path: String,
    format: String,
    coordinate_space: String,
    sha256: String,
    byte_length: u64,
    vertex_count: u64,
    triangle_count: u64,
    algorithm_version: u32,
}

impl OverviewMesh {
    /// Load one immutable overview. Missing legacy metadata is a local error.
    pub fn load(manifest_path: &Path, expected_identity: &str) -> Result<Self, String> {
        let text = fs::read_to_string(manifest_path)
            .map_err(|error| format!("overview manifest unavailable: {error}"))?;
        let manifest: Manifest = serde_json::from_str(&text)
            .map_err(|error| format!("overview manifest invalid: {error}"))?;
        if manifest.identity_sha256 != expected_identity {
            return Err("overview world identity mismatch".to_string());
        }
        let entry = manifest
            .overview
            .ok_or_else(|| "overview unavailable for this legacy compiled world".to_string())?;
        if entry.schema_version != 1
            || entry.format != "indexed_u32_le"
            || entry.coordinate_space != "storage"
            || entry.algorithm_version != SUPPORTED_ALGORITHM_VERSION
        {
            return Err("overview format is unsupported".to_string());
        }
        if entry.relative_path != "overview.mesh" {
            return Err("overview path is unsupported".to_string());
        }
        let root = manifest_path
            .parent()
            .ok_or("overview manifest has no parent")?;
        let payload = fs::read(root.join(&entry.relative_path))
            .map_err(|error| format!("overview mesh unavailable: {error}"))?;
        if payload.len() as u64 != entry.byte_length {
            return Err("overview mesh byte length mismatch".to_string());
        }
        if format!("{:x}", Sha256::digest(&payload)) != entry.sha256 {
            return Err("overview mesh checksum mismatch".to_string());
        }
        let mesh = decode(
            &payload,
            [manifest.extent.x, manifest.extent.y, manifest.extent.z],
        )?;
        if mesh.vertices.len() as u64 != entry.vertex_count
            || mesh.indices.len() as u64 / 3 != entry.triangle_count
        {
            return Err("overview mesh metadata mismatch".to_string());
        }
        Ok(mesh)
    }

    /// Project after f64 centering/scaling, before conversion to f32.
    pub fn project(&self, yaw: f32, pitch: f32) -> Vec<ProjectedVertex> {
        let center = self.extent.map(|value| f64::from(value) * 0.5);
        let scale = f64::from(*self.extent.iter().max().unwrap_or(&1)).max(1.0);
        let (sy, cy) = f64::from(yaw).sin_cos();
        let (sp, cp) = f64::from(pitch).sin_cos();
        self.vertices
            .iter()
            .map(|vertex| {
                let x = (f64::from(vertex[0]) - center[0]) / scale;
                let y = (f64::from(vertex[1]) - center[1]) / scale;
                let z = (f64::from(vertex[2]) - center[2]) / scale;
                // Voxel worlds are Z-up: yaw rotates the horizontal X/Y
                // plane, then pitch tilts height Z against horizontal depth.
                let xr = x * cy - y * sy;
                let horizontal_depth = x * sy + y * cy;
                let vertical = z * cp - horizontal_depth * sp;
                let depth = z * sp + horizontal_depth * cp;
                ProjectedVertex {
                    x: xr as f32,
                    y: -vertical as f32,
                    depth: depth as f32,
                }
            })
            .collect()
    }

    pub fn bounds_vertices(&self) -> Vec<[u32; 3]> {
        let [x, y, z] = self.extent;
        vec![
            [0, 0, 0],
            [x, 0, 0],
            [x, y, 0],
            [0, y, 0],
            [0, 0, z],
            [x, 0, z],
            [x, y, z],
            [0, y, z],
        ]
    }
}

fn decode(payload: &[u8], extent: [u32; 3]) -> Result<OverviewMesh, String> {
    if payload.len() < HEADER_BYTES || &payload[..4] != MAGIC {
        return Err("overview mesh header invalid".to_string());
    }
    let word = |offset| u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
    if word(4) != SUPPORTED_ALGORITHM_VERSION {
        return Err("overview mesh algorithm unsupported".to_string());
    }
    let vertex_count = usize::try_from(word(8)).map_err(|_| "vertex count overflow")?;
    let index_count = usize::try_from(word(12)).map_err(|_| "index count overflow")?;
    if index_count % 3 != 0 {
        return Err("overview mesh indices are not triangular".to_string());
    }
    let expected = HEADER_BYTES
        .checked_add(
            vertex_count
                .checked_mul(12)
                .ok_or("vertex bytes overflow")?,
        )
        .and_then(|value| value.checked_add(index_count.checked_mul(4)?))
        .ok_or("overview mesh size overflow")?;
    if payload.len() != expected {
        return Err("overview mesh structure invalid".to_string());
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    let mut offset = HEADER_BYTES;
    for _ in 0..vertex_count {
        let vertex = [word(offset), word(offset + 4), word(offset + 8)];
        if vertex
            .iter()
            .zip(extent)
            .any(|(value, limit)| *value > limit)
        {
            return Err("overview vertex outside declared extent".to_string());
        }
        vertices.push(vertex);
        offset += 12;
    }
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        let index = word(offset);
        if index as usize >= vertex_count {
            return Err("overview index outside vertex array".to_string());
        }
        indices.push(index);
        offset += 4;
    }
    Ok(OverviewMesh {
        vertices,
        indices,
        extent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn payload(vertices: &[[u32; 3]], indices: &[u32]) -> Vec<u8> {
        let mut result = Vec::from(MAGIC.as_slice());
        for value in [1, vertices.len() as u32, indices.len() as u32] {
            result.extend(value.to_le_bytes());
        }
        for vertex in vertices {
            for value in vertex {
                result.extend(value.to_le_bytes());
            }
        }
        for index in indices {
            result.extend(index.to_le_bytes());
        }
        result
    }

    #[test]
    fn decodes_bounded_indexed_mesh() {
        let mesh = decode(
            &payload(&[[0, 0, 0], [10, 0, 0], [0, 10, 0]], &[0, 1, 2]),
            [10, 10, 10],
        )
        .unwrap();
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn rejects_invalid_index_and_extent() {
        assert!(decode(&payload(&[[0, 0, 0]], &[0, 0, 1]), [1, 1, 1]).is_err());
        assert!(decode(&payload(&[[2, 0, 0]], &[]), [1, 1, 1]).is_err());
    }

    #[test]
    fn projection_preserves_elongated_world_aspect_and_large_coordinate_delta() {
        let mesh = OverviewMesh {
            vertices: vec![[60_000, 20_000, 10_000], [59_999, 20_000, 10_000]],
            indices: vec![],
            extent: [60_000, 40_000, 20_000],
        };
        let points = mesh.project(0.0, 0.0);
        assert!(points[0].x > points[1].x);
        assert!((points[0].x - points[1].x - 1.0 / 60_000.0).abs() < 1e-6);
    }

    #[test]
    fn projection_uses_z_as_vertical_axis() {
        let mesh = OverviewMesh {
            vertices: vec![[5, 5, 5], [5, 5, 6], [5, 6, 5]],
            indices: vec![],
            extent: [10, 10, 10],
        };
        let points = mesh.project(0.0, 0.0);
        assert!(points[1].y < points[0].y);
        assert_eq!(points[2].y, points[0].y);
        assert!(points[2].depth > points[0].depth);
    }

    #[test]
    fn missing_and_corrupt_overviews_are_local_load_errors() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("theseo-overview-{unique}"));
        fs::create_dir(&root).unwrap();
        let manifest_path = root.join("manifest.json");
        fs::write(
            &manifest_path,
            r#"{"identity_sha256":"id","extent":{"x":10,"y":10,"z":10}}"#,
        )
        .unwrap();
        assert!(OverviewMesh::load(&manifest_path, "id")
            .unwrap_err()
            .contains("legacy"));

        let bytes = payload(&[[0, 0, 0]], &[]);
        fs::write(root.join("overview.mesh"), &bytes).unwrap();
        fs::write(
            &manifest_path,
            format!(
                r#"{{"identity_sha256":"id","extent":{{"x":10,"y":10,"z":10}},"overview":{{"schema_version":1,"relative_path":"overview.mesh","format":"indexed_u32_le","coordinate_space":"storage","sha256":"{}","byte_length":{},"vertex_count":1,"triangle_count":0,"algorithm_version":1}}}}"#,
                "0".repeat(64), bytes.len()
            ),
        ).unwrap();
        assert!(OverviewMesh::load(&manifest_path, "id")
            .unwrap_err()
            .contains("checksum"));
        fs::remove_dir_all(root).unwrap();
    }
}
