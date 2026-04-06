/// Pool Explorer — browse a pre-computed geometry pool directory.
///
/// Usage:
///   pool-explorer <pool_dir>          — browse the whole pool
///   pool-explorer <entry.npy>         — view a single pool entry
///
/// Left panel:  scrollable list of all .npy entries, grouped by source,
///              showing fill % and a small bar per entry.
/// Right panel: isometric 3D view of the selected geometry.
///              Gray  — filled voxels (obstacle geometry)
///              Teal  — boundary of empty space (free cells 6-adjacent to filled)
///              Pink  — original STL mesh triangles (toggle with S)
///
/// Controls:
///   Left-drag   — orbit camera
///   Scroll      — zoom
///   R           — reset camera
///   ↑ / ↓      — previous / next entry
///   S           — toggle original STL mesh overlay
use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Color32, Key, Pos2, Rect, ScrollArea, Sense, Shape, Stroke, Vec2,
};

// ---------------------------------------------------------------------------
// .npy reader — handles uint8 (|u1) 3-D cubic arrays, v1 and v2 format
// ---------------------------------------------------------------------------

fn parse_npy_uint8_3d(bytes: &[u8]) -> Result<(Vec<u8>, usize), String> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a numpy file".into());
    }
    let major = bytes[6];
    let (header_len, hdr_offset) = if major == 1 {
        (u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize, 10)
    } else {
        if bytes.len() < 12 {
            return Err("truncated v2 header".into());
        }
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let hdr_end = hdr_offset + header_len;
    if bytes.len() < hdr_end {
        return Err("truncated header data".into());
    }
    let header = std::str::from_utf8(&bytes[hdr_offset..hdr_end])
        .map_err(|e| format!("header utf-8: {e}"))?;

    // Parse the shape tuple from the header dict.
    let shape_pos = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or("no 'shape' key in header")?;
    let after = &header[shape_pos..];
    let tuple_start = after.find('(').ok_or("no '(' in shape")? + shape_pos;
    let tuple_end = header[tuple_start..].find(')').ok_or("no ')' in shape")? + tuple_start;
    let tuple_str = &header[tuple_start + 1..tuple_end];
    let dims: Vec<usize> = tuple_str
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 3 {
        return Err(format!("expected 3D array, got {} dimensions", dims.len()));
    }
    let g = dims[0];
    let expected = dims[0] * dims[1] * dims[2];
    let data_bytes = &bytes[hdr_end..];
    if data_bytes.len() < expected {
        return Err(format!("data too short: {} < {}", data_bytes.len(), expected));
    }
    Ok((data_bytes[..expected].to_vec(), g))
}

// ---------------------------------------------------------------------------
// ASCII STL mesh — triangles normalised to [1, grid_size]³
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Triangle {
    v: [[f32; 3]; 3],
}

/// Parse an ASCII STL and return triangles normalised so the longest axis
/// spans `target_span` voxels starting at `origin` on each axis.
/// This matches the same normalisation used by `_load_stl_geometry` in Python.
fn parse_rotation_matrix(v: &serde_json::Value) -> Option<[[f32; 3]; 3]> {
    let arr = v.as_array()?;
    if arr.len() != 3 { return None; }
    let mut m = [[0.0f32; 3]; 3];
    for (i, row) in arr.iter().enumerate() {
        let row = row.as_array()?;
        if row.len() != 3 { return None; }
        for (j, val) in row.iter().enumerate() {
            m[i][j] = val.as_f64()? as f32;
        }
    }
    Some(m)
}

fn rotate_vertex(v: [f32; 3], r: &[[f32; 3]; 3]) -> [f32; 3] {
    [
        r[0][0]*v[0] + r[0][1]*v[1] + r[0][2]*v[2],
        r[1][0]*v[0] + r[1][1]*v[1] + r[1][2]*v[2],
        r[2][0]*v[0] + r[2][1]*v[1] + r[2][2]*v[2],
    ]
}

/// Parse and normalise an ASCII STL into the same grid coordinate space used by voxelization.
/// `scale` matches the per-entry scale from the .meta.json sidecar (None → fill max_span).
/// `rotation` matches the rotation matrix applied before voxelization (None → no rotation).
fn parse_stl_normalised(
    path: &Path,
    grid_size: usize,
    padding: usize,
    scale: Option<f32>,
    rotation: Option<[[f32; 3]; 3]>,
) -> Vec<Triangle> {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    // Collect all vertices
    let mut raw_verts: Vec<[f32; 3]> = Vec::new();
    for line in content.lines() {
        let t = line.trim();
        if t.starts_with("vertex") {
            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() >= 4 {
                if let (Ok(x), Ok(y), Ok(z)) = (
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                    parts[3].parse::<f32>(),
                ) {
                    raw_verts.push([x, y, z]);
                }
            }
        }
    }
    if raw_verts.is_empty() {
        return vec![];
    }

    // Apply rotation (same matrix that was applied before voxelization)
    if let Some(r) = rotation {
        for v in &mut raw_verts {
            *v = rotate_vertex(*v, &r);
        }
    }

    // Bounding box (after rotation, matching Python's _stl_bounding_box)
    let mut min = raw_verts[0];
    let mut max = raw_verts[0];
    for v in &raw_verts {
        for i in 0..3 {
            if v[i] < min[i] { min[i] = v[i]; }
            if v[i] > max[i] { max[i] = v[i]; }
        }
    }
    let extents = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let max_extent = extents.iter().cloned().fold(0.0_f32, f32::max).max(1e-6);

    // Same normalisation as _load_stl_geometry:
    // effective_scale = min(entry_scale, max_span); vpu = effective_scale / max_extent
    let max_span = (grid_size - 2 * padding - 1) as f32;
    let effective_scale = scale.map(|s| s.min(max_span)).unwrap_or(max_span);
    let vpu = effective_scale / max_extent;
    let origin = (padding + 1) as f32;
    let adj = [
        origin - min[0] * vpu,
        origin - min[1] * vpu,
        origin - min[2] * vpu,
    ];
    let norm = |v: [f32; 3]| -> [f32; 3] {
        [v[0] * vpu + adj[0], v[1] * vpu + adj[1], v[2] * vpu + adj[2]]
    };

    // Rebuild triangles (every 3 vertices is one triangle)
    raw_verts.chunks(3)
        .filter(|c| c.len() == 3)
        .map(|c| Triangle { v: [norm(c[0]), norm(c[1]), norm(c[2])] })
        .collect()
}

/// Draw a single STL triangle in the isometric view.
fn draw_triangle(
    painter: &egui::Painter,
    tri: &Triangle,
    rect: Rect,
    cam: &Camera,
    b: &Bounds,
    fill: Color32,
    stroke: Stroke,
) {
    let pts: Vec<Pos2> = tri.v.iter()
        .map(|&[x, y, z]| cam.to_screen(x, y, z, rect, b))
        .collect();
    painter.add(Shape::convex_polygon(pts, fill, stroke));
}

// ---------------------------------------------------------------------------
// Geometry derived from a loaded grid
// ---------------------------------------------------------------------------

/// Filled cells that have at least one free 6-neighbour (= visible surface).
fn surface_filled(grid: &[u8], g: usize) -> Vec<(u16, u16, u16)> {
    let idx = |x: usize, y: usize, z: usize| x * g * g + y * g + z;
    const DIRS: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ];
    let mut out = Vec::new();
    for x in 0..g {
        for y in 0..g {
            for z in 0..g {
                if grid[idx(x, y, z)] == 0 {
                    continue;
                }
                let exposed = DIRS.iter().any(|&(dx, dy, dz)| {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= g as i32
                        || ny >= g as i32
                        || nz >= g as i32
                        || grid[idx(nx as usize, ny as usize, nz as usize)] == 0
                });
                if exposed {
                    // +1: convert 0-based array index to 1-based env coord
                    out.push((x as u16 + 1, y as u16 + 1, z as u16 + 1));
                }
            }
        }
    }
    out
}

/// Free cells 6-adjacent to at least one filled cell (= boundary of empty space).
fn surface_free(grid: &[u8], g: usize) -> Vec<(u16, u16, u16)> {
    let idx = |x: usize, y: usize, z: usize| x * g * g + y * g + z;
    const DIRS: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ];
    let mut out = Vec::new();
    for x in 0..g {
        for y in 0..g {
            for z in 0..g {
                if grid[idx(x, y, z)] != 0 {
                    continue;
                }
                let adj_filled = DIRS.iter().any(|&(dx, dy, dz)| {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    nx >= 0
                        && ny >= 0
                        && nz >= 0
                        && nx < g as i32
                        && ny < g as i32
                        && nz < g as i32
                        && grid[idx(nx as usize, ny as usize, nz as usize)] != 0
                });
                if adj_filled {
                    out.push((x as u16 + 1, y as u16 + 1, z as u16 + 1));
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Entry list
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct PoolEntry {
    path: PathBuf,
    source: String,         // sub-directory name
    filename: String,       // e.g. "0002.npy"
    fill_pct: f32,
    filled_count: usize,
    grid_size: usize,
    source_stl: Option<PathBuf>,          // original STL path from pool_meta.json
    entry_scale: Option<f32>,             // exact scale used during voxelization
    entry_rotation: Option<[[f32; 3]; 3]>, // exact rotation matrix used (None = no rotation)
}

fn scan_pool(pool_dir: &Path) -> Vec<PoolEntry> {
    let mut entries: Vec<PoolEntry> = Vec::new();

    // If we were given a single .npy file, treat it as a one-entry pool
    if pool_dir.extension().and_then(|e| e.to_str()) == Some("npy") {
        if let Ok(bytes) = std::fs::read(pool_dir) {
            if let Ok((grid, g)) = parse_npy_uint8_3d(&bytes) {
                let filled = grid.iter().filter(|&&v| v != 0).count();
                let fill_pct = 100.0 * filled as f32 / (g * g * g) as f32;
                let filename = pool_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string();
                let source = pool_dir
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string();
                // Try to find pool_meta.json one level up
                let source_stl = pool_dir.parent()
                    .and_then(|p| p.parent())
                    .map(|d| d.join("pool_meta.json"))
                    .and_then(|m| std::fs::read_to_string(m).ok())
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v["source_paths"][&source].as_str().map(PathBuf::from));
                let sidecar: Option<serde_json::Value> = std::fs::read_to_string(pool_dir.with_extension("meta.json"))
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
                let entry_scale = sidecar.as_ref().and_then(|v| v["scale"].as_f64()).map(|s| s as f32);
                let entry_rotation = sidecar.as_ref().and_then(|v| parse_rotation_matrix(&v["rotation"]));
                entries.push(PoolEntry {
                    path: pool_dir.to_path_buf(),
                    source,
                    filename,
                    fill_pct,
                    filled_count: filled,
                    grid_size: g,
                    source_stl,
                    entry_scale,
                    entry_rotation,
                });
            }
        }
        return entries;
    }

    // Read source_paths from pool_meta.json (may be absent for old pools)
    let meta: serde_json::Value = std::fs::read_to_string(pool_dir.join("pool_meta.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .unwrap_or_default();
    let source_paths = meta.get("source_paths");

    // Scan sub-directories
    let mut sources: Vec<String> = std::fs::read_dir(pool_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    sources.sort();

    for source in sources {
        let src_dir = pool_dir.join(&source);
        let mut npy_files: Vec<PathBuf> = std::fs::read_dir(&src_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("npy"))
            .collect();
        npy_files.sort();

        // Only treat the source path as an STL if it has a .stl extension
        let source_stl: Option<PathBuf> = source_paths
            .and_then(|sp| sp[&source].as_str())
            .map(PathBuf::from)
            .filter(|p: &PathBuf| {
                p.exists()
                    && p.extension().and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("stl"))
                        .unwrap_or(false)
            });

        for path in npy_files {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string();
            let (fill_pct, filled_count, grid_size) = std::fs::read(&path)
                .ok()
                .and_then(|b| parse_npy_uint8_3d(&b).ok())
                .map(|(grid, g)| {
                    let filled = grid.iter().filter(|&&v| v != 0).count();
                    (100.0 * filled as f32 / (g * g * g) as f32, filled, g)
                })
                .unwrap_or((0.0, 0, 32));

            // Read per-entry sidecar: scale and rotation used during voxelization
            let sidecar: Option<serde_json::Value> = std::fs::read_to_string(path.with_extension("meta.json"))
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
            let entry_scale = sidecar.as_ref().and_then(|v| v["scale"].as_f64()).map(|s| s as f32);
            let entry_rotation = sidecar.as_ref().and_then(|v| parse_rotation_matrix(&v["rotation"]));

            entries.push(PoolEntry {
                path,
                source: source.clone(),
                filename,
                fill_pct,
                filled_count,
                grid_size,
                source_stl: source_stl.clone(),
                entry_scale,
                entry_rotation,
            });
        }
    }
    entries
}

// Helper trait to make piping cleaner
trait Pipe: Sized {
    fn pipe<F: FnOnce(Self) -> R, R>(self, f: F) -> R { f(self) }
}

// ---------------------------------------------------------------------------
// Loaded geometry (expensive: computed once per selected entry)
// ---------------------------------------------------------------------------

struct LoadedGeometry {
    /// Visible surface of filled cells — the only cells we need to render
    filled_surface: Vec<(u16, u16, u16)>,
    /// Free cells adjacent to filled cells — the navigable boundary
    free_surface: Vec<(u16, u16, u16)>,
    /// Original STL triangles, normalised to the same grid space (optional)
    stl_triangles: Vec<Triangle>,
    /// Depth-sorted index into stl_triangles (rebuilt when camera changes)
    stl_sorted: Vec<usize>,
    grid_size: usize,
    fill_pct: f32,
    filled_count: usize,
    free_count: usize,
    free_surface_count: usize,
}

impl LoadedGeometry {
    fn load(entry: &PoolEntry) -> Option<Self> {
        let bytes = std::fs::read(&entry.path).ok()?;
        let (grid, g) = parse_npy_uint8_3d(&bytes).ok()?;
        let total = g * g * g;
        let filled_count = grid.iter().filter(|&&v| v != 0).count();
        let free_count = total - filled_count;
        let fill_pct = 100.0 * filled_count as f32 / total as f32;
        // All filled cells — no culling so maps show their full solid structure
        let filled_surface: Vec<(u16, u16, u16)> = grid.iter().enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(idx, _)| {
                let ix = idx / (g * g);
                let iy = (idx / g) % g;
                let iz = idx % g;
                (ix as u16 + 1, iy as u16 + 1, iz as u16 + 1)
            })
            .collect();
        let free_surface = surface_free(&grid, g);
        let free_surface_count = free_surface.len();

        let stl_triangles = entry.source_stl.as_deref()
            .map(|p| parse_stl_normalised(p, g, 2, entry.entry_scale, entry.entry_rotation))
            .unwrap_or_default();
        let stl_sorted = (0..stl_triangles.len()).collect();

        Some(Self {
            filled_surface,
            free_surface,
            stl_triangles,
            stl_sorted,
            grid_size: g,
            fill_pct,
            filled_count,
            free_count,
            free_surface_count,
        })
    }

    fn sort_stl(&mut self, cam: &Camera) {
        let tris = &self.stl_triangles;
        self.stl_sorted.sort_by(|&a, &b| {
            let depth = |i: usize| -> f32 {
                let v = &tris[i].v;
                let cx = (v[0][0] + v[1][0] + v[2][0]) / 3.0;
                let cy = (v[0][1] + v[1][1] + v[2][1]) / 3.0;
                let cz = (v[0][2] + v[1][2] + v[2][2]) / 3.0;
                let (sy, cy_) = (cam.yaw.sin(), cam.yaw.cos());
                let (sp, cp) = (cam.pitch.sin(), cam.pitch.cos());
                cx * sy * cp + cy * sp + cz * cy_ * cp
            };
            depth(a).partial_cmp(&depth(b)).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// ---------------------------------------------------------------------------
// Camera — identical to voxel_replay
// ---------------------------------------------------------------------------

struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

impl Camera {
    fn default() -> Self {
        Self {
            yaw: 45.0_f32.to_radians(),
            pitch: 30.0_f32.to_radians(),
            zoom: 1.0,
        }
    }

    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let xr = x * self.yaw.cos() - z * self.yaw.sin();
        let zr = x * self.yaw.sin() + z * self.yaw.cos();
        let yr = y * self.pitch.cos() - zr * self.pitch.sin();
        let zr2 = y * self.pitch.sin() + zr * self.pitch.cos();
        (xr, -(yr - zr2 * 0.05))
    }

    fn bounds(&self, g: f32) -> Bounds {
        let lo = 0.5_f32;
        let hi = g + 0.5;
        let (mut min_x, mut max_x, mut min_y, mut max_y) =
            (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for &xf in &[lo, hi] {
            for &yf in &[lo, hi] {
                for &zf in &[lo, hi] {
                    let (px, py) = self.project(xf, yf, zf);
                    if px < min_x { min_x = px; }
                    if px > max_x { max_x = px; }
                    if py < min_y { min_y = py; }
                    if py > max_y { max_y = py; }
                }
            }
        }
        let pw = (max_x - min_x) * 0.05;
        let ph = (max_y - min_y) * 0.05;
        Bounds { min_x: min_x - pw, max_x: max_x + pw, min_y: min_y - ph, max_y: max_y + ph }
    }

    fn to_screen(&self, x: f32, y: f32, z: f32, rect: Rect, b: &Bounds) -> Pos2 {
        let (px, py) = self.project(x, y, z);
        let w = rect.width() * self.zoom;
        let h = rect.height() * self.zoom;
        Pos2::new(
            rect.center().x
                + (px - (b.min_x + b.max_x) * 0.5) / (b.max_x - b.min_x).max(1.0) * w,
            rect.center().y
                + (py - (b.min_y + b.max_y) * 0.5) / (b.max_y - b.min_y).max(1.0) * h,
        )
    }
}

struct Bounds {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

// ---------------------------------------------------------------------------
// Voxel rendering helpers
// ---------------------------------------------------------------------------

fn depth_key(x: u16, y: u16, z: u16, cam: &Camera) -> f32 {
    let (sy, cy) = (cam.yaw.sin(), cam.yaw.cos());
    let (sp, cp) = (cam.pitch.sin(), cam.pitch.cos());
    x as f32 * sy * cp + y as f32 * sp + z as f32 * cy * cp
}

fn draw_voxel(
    painter: &egui::Painter,
    cx: u16, cy: u16, cz: u16,
    rect: Rect, cam: &Camera, b: &Bounds,
    base: Color32,
    outline: bool,
) {
    let (x, y, z) = (cx as f32, cy as f32, cz as f32);
    let h = 0.5_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);

    let hx = if cam.yaw.sin() > 0.0 { h } else { -h };
    let hy = if cam.pitch.sin() > 0.0 { h } else { -h };
    let hz = if cam.yaw.cos() > 0.0 { h } else { -h };

    let shade = |r: u8, g: u8, bl: u8, amt: i32| -> Color32 {
        Color32::from_rgb(
            (r as i32 + amt).clamp(0, 255) as u8,
            (g as i32 + amt).clamp(0, 255) as u8,
            (bl as i32 + amt).clamp(0, 255) as u8,
        )
    };
    let (r, g, bl) = (base.r(), base.g(), base.b());
    let stroke = if outline { Stroke::new(0.4, Color32::from_gray(20)) } else { Stroke::NONE };

    painter.add(Shape::convex_polygon(
        vec![corner(-h, -h, hz), corner(h, -h, hz), corner(h, h, hz), corner(-h, h, hz)],
        shade(r, g, bl, -40), stroke,
    ));
    painter.add(Shape::convex_polygon(
        vec![corner(hx, -h, -h), corner(hx, -h, h), corner(hx, h, h), corner(hx, h, -h)],
        shade(r, g, bl, -10), stroke,
    ));
    painter.add(Shape::convex_polygon(
        vec![corner(-h, hy, -h), corner(h, hy, -h), corner(h, hy, h), corner(-h, hy, h)],
        shade(r, g, bl, 40), stroke,
    ));
}

fn draw_wireframe_box(painter: &egui::Painter, rect: Rect, cam: &Camera, b: &Bounds, g: f32) {
    let stroke = Stroke::new(1.0, Color32::from_rgba_premultiplied(80, 80, 80, 60));
    let lo = 0.5_f32;
    let hi = g + 0.5;
    let corners = [
        (lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo),
        (lo, lo, hi), (hi, lo, hi), (hi, hi, hi), (lo, hi, hi),
    ];
    let edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];
    let pts: Vec<Pos2> = corners.iter().map(|&(x,y,z)| cam.to_screen(x, y, z, rect, b)).collect();
    for (a, bi) in edges {
        painter.line_segment([pts[a], pts[bi]], stroke);
    }
}

// ---------------------------------------------------------------------------
// Fill level bar
// ---------------------------------------------------------------------------

fn draw_fill_bar(ui: &mut egui::Ui, fill_pct: f32, width: f32) {
    let bar_h = 6.0_f32;
    let (resp, painter) = ui.allocate_painter(Vec2::new(width, bar_h), Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, 2.0, Color32::from_gray(45));
    let fill_w = (fill_pct / 100.0).clamp(0.0, 1.0) * r.width();
    let col = if fill_pct < 10.0 {
        Color32::from_rgb(60, 180, 100) // low fill (mostly navigable) → green
    } else if fill_pct < 40.0 {
        Color32::from_rgb(200, 180, 40) // medium → yellow
    } else {
        Color32::from_rgb(210, 80, 50)  // high fill (dense obstacles) → red
    };
    painter.rect_filled(
        Rect::from_min_size(r.min, Vec2::new(fill_w, bar_h)),
        2.0, col,
    );
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct PoolExplorerApp {
    pool_dir: String,
    entries: Vec<PoolEntry>,
    selected: usize,
    geometry: Option<LoadedGeometry>,
    camera: Camera,
    show_stl: bool,
    last_yaw: f32,  // detect camera change to re-sort STL triangles
}

impl PoolExplorerApp {
    fn new(pool_dir: &Path, entries: Vec<PoolEntry>) -> Self {
        let mut app = Self {
            pool_dir: pool_dir.display().to_string(),
            entries,
            selected: 0,
            geometry: None,
            camera: Camera::default(),
            show_stl: false,
            last_yaw: f32::NAN,
        };
        app.load_selected();
        app
    }

    fn load_selected(&mut self) {
        self.geometry = self.entries.get(self.selected).and_then(LoadedGeometry::load);
    }
}

impl eframe::App for PoolExplorerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Keyboard navigation ---
        let prev = ctx.input(|i| i.key_pressed(Key::ArrowUp));
        let next = ctx.input(|i| i.key_pressed(Key::ArrowDown));
        let reset_cam = ctx.input(|i| i.key_pressed(Key::R));
        let toggle_stl = ctx.input(|i| i.key_pressed(Key::S));

        if prev && self.selected > 0 {
            self.selected -= 1;
            self.load_selected();
        }
        if next && self.selected + 1 < self.entries.len() {
            self.selected += 1;
            self.load_selected();
        }
        if reset_cam {
            self.camera = Camera::default();
        }
        if toggle_stl {
            self.show_stl = !self.show_stl;
        }

        // --- Camera drag / scroll ---
        let drag_delta = ctx.input(|i| i.pointer.delta());
        let dragging = ctx.input(|i| i.pointer.primary_down());
        let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);

        if dragging {
            self.camera.yaw += drag_delta.x * 0.005;
            self.camera.pitch = (self.camera.pitch - drag_delta.y * 0.005)
                .clamp(-1.5, 1.5);
        }
        if scroll_y != 0.0 {
            self.camera.zoom = (self.camera.zoom * (1.0 + scroll_y * 0.001)).clamp(0.2, 8.0);
        }

        // ---- Left panel: scrollable entry list ----
        egui::SidePanel::left("pool_list")
            .exact_width(270.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.heading("Geometry Pool");
                ui.label(egui::RichText::new(&self.pool_dir).small().weak());
                ui.label(
                    egui::RichText::new(format!("{} entries", self.entries.len()))
                        .small()
                        .weak(),
                );
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let voxel_label = egui::RichText::new("Voxel").strong();
                    let stl_label   = egui::RichText::new("STL").strong();
                    if ui.add(egui::SelectableLabel::new(!self.show_stl, voxel_label)).clicked() {
                        self.show_stl = false;
                    }
                    if ui.add(egui::SelectableLabel::new(self.show_stl, stl_label)).clicked() {
                        self.show_stl = true;
                    }
                    ui.label(egui::RichText::new("(or press S)").small().weak());
                });
                ui.separator();

                ScrollArea::vertical().show(ui, |ui| {
                    let mut current_source = String::new();
                    let mut clicked: Option<usize> = None;

                    for (i, entry) in self.entries.iter().enumerate() {
                        // Source group header
                        if entry.source != current_source {
                            if !current_source.is_empty() {
                                ui.add_space(4.0);
                            }
                            ui.label(
                                egui::RichText::new(&entry.source)
                                    .strong()
                                    .color(Color32::from_rgb(140, 180, 220)),
                            );
                            current_source = entry.source.clone();
                        }

                        let is_sel = i == self.selected;
                        let bg = if is_sel {
                            Color32::from_rgb(40, 55, 75)
                        } else {
                            Color32::TRANSPARENT
                        };

                        let item_resp = ui.push_id(i, |ui| {
                            egui::Frame::NONE
                                .fill(bg)
                                .inner_margin(egui::Margin::symmetric(6, 3))
                                .corner_radius(4)
                                .show(ui, |ui| {
                                    ui.set_min_width(240.0);
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(&entry.filename)
                                                .monospace()
                                                .small(),
                                        );
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "{:.1}%",
                                                        entry.fill_pct
                                                    ))
                                                    .small()
                                                    .color(fill_color(entry.fill_pct)),
                                                );
                                            },
                                        );
                                    });
                                    draw_fill_bar(ui, entry.fill_pct, 238.0);
                                })
                        });

                        if item_resp.response.interact(Sense::click()).clicked() {
                            clicked = Some(i);
                        }
                    }

                    if let Some(idx) = clicked {
                        if self.selected != idx {
                            self.selected = idx;
                            self.load_selected();
                        }
                    }
                });
            });

        // ---- Central panel: 3D view ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::drag());
            let rect = resp.rect.shrink(16.0);

            painter.rect_filled(resp.rect, 0.0, Color32::from_rgb(10, 12, 18));

            // Re-sort STL triangles when camera yaw changes (painter's algorithm).
            // Must happen before the immutable borrow of self.geometry below.
            if (self.camera.yaw - self.last_yaw).abs() > 0.001 {
                if let Some(geo) = self.geometry.as_mut() {
                    geo.sort_stl(&self.camera);
                }
                self.last_yaw = self.camera.yaw;
            }

            let Some(geo) = &self.geometry else {
                painter.text(
                    resp.rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "No geometry loaded",
                    egui::FontId::proportional(16.0),
                    Color32::from_gray(100),
                );
                return;
            };

            let g = geo.grid_size as f32;
            let cam = &self.camera;
            let b = cam.bounds(g);

            // Draw bounding box wireframe
            draw_wireframe_box(&painter, rect, cam, &b, g);

            let geo_color = Color32::from_rgb(110, 115, 125);
            let boundary_color = Color32::from_rgb(50, 190, 175);
            let stl_color = Color32::from_rgb(220, 130, 170);

            if self.show_stl {
                // ---- STL mesh view ----
                if geo.stl_triangles.is_empty() {
                    painter.text(
                        resp.rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "No source STL path in pool_meta.json\nRe-run: anysearch extract",
                        egui::FontId::proportional(12.0),
                        Color32::from_rgb(220, 120, 80),
                    );
                } else {
                    let stl_fill = Color32::from_rgba_premultiplied(180, 90, 130, 200);
                    let stl_stroke = Stroke::new(0.5, Color32::from_rgb(240, 160, 200));
                    for &ti in &geo.stl_sorted {
                        draw_triangle(&painter, &geo.stl_triangles[ti], rect, cam, &b, stl_fill, stl_stroke);
                    }
                }
            } else {
                // ---- Voxel view ----
                // Sort filled surface back-to-front and render (gray)
                let mut filled_sorted = geo.filled_surface.clone();
                filled_sorted.sort_by(|a, c| {
                    depth_key(a.0, a.1, a.2, cam)
                        .partial_cmp(&depth_key(c.0, c.1, c.2, cam))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                // Limit draw count for large grids to keep UI responsive
                let filled_limit = filled_sorted.len().min(8000);
                let step = if filled_sorted.len() > filled_limit {
                    filled_sorted.len() / filled_limit
                } else {
                    1
                };
                for chunk in filled_sorted.chunks(step) {
                    let (x, y, z) = chunk[0];
                    draw_voxel(&painter, x, y, z, rect, cam, &b, geo_color, false);
                }

                // Sort free surface back-to-front and render (teal)
                let mut free_sorted = geo.free_surface.clone();
                free_sorted.sort_by(|a, c| {
                    depth_key(a.0, a.1, a.2, cam)
                        .partial_cmp(&depth_key(c.0, c.1, c.2, cam))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let free_limit = free_sorted.len().min(8000);
                let free_step = if free_sorted.len() > free_limit {
                    free_sorted.len() / free_limit
                } else {
                    1
                };
                for chunk in free_sorted.chunks(free_step) {
                    let (x, y, z) = chunk[0];
                    draw_voxel(&painter, x, y, z, rect, cam, &b, boundary_color, false);
                }
            }

            // ---- Stats overlay (top-left) ----
            let entry = &self.entries[self.selected];
            let stats = format!(
                "{} / {}\n{}\ngrid {}³   filled {}  ({:.1}%)\nboundary {} free  /  {} free surface",
                entry.source,
                entry.filename,
                self.pool_dir,
                geo.grid_size,
                geo.filled_count,
                geo.fill_pct,
                geo.free_surface_count,
                geo.free_count,
            );
            painter.text(
                resp.rect.min + Vec2::new(12.0, 10.0),
                egui::Align2::LEFT_TOP,
                &stats,
                egui::FontId::monospace(11.0),
                Color32::from_gray(180),
            );

            // ---- Fill level bar (top-right) ----
            let bar_w = 140.0_f32;
            let bar_h = 10.0_f32;
            let bar_x = resp.rect.right() - bar_w - 12.0;
            let bar_y = resp.rect.top() + 12.0;
            let bar_rect =
                Rect::from_min_size(Pos2::new(bar_x, bar_y), Vec2::new(bar_w, bar_h));
            painter.rect_filled(bar_rect, 3.0, Color32::from_gray(40));
            let fill_w = (geo.fill_pct / 100.0).clamp(0.0, 1.0) * bar_w;
            painter.rect_filled(
                Rect::from_min_size(bar_rect.min, Vec2::new(fill_w, bar_h)),
                3.0,
                fill_color(geo.fill_pct),
            );
            painter.text(
                Pos2::new(bar_x + bar_w * 0.5, bar_y + bar_h * 0.5),
                egui::Align2::CENTER_CENTER,
                format!("{:.1}% filled", geo.fill_pct),
                egui::FontId::proportional(10.0),
                Color32::WHITE,
            );

            // ---- Legend (bottom-right) ----
            let lx = resp.rect.right() - 175.0;
            let legend_items: &[(Color32, &str)] = if self.show_stl {
                &[(stl_color, "original STL mesh")]
            } else {
                &[
                    (geo_color, "filled (obstacle)"),
                    (boundary_color, "boundary (navigable edge)"),
                ]
            };
            let legend_h = 14.0 + legend_items.len() as f32 * 18.0;
            let ly = resp.rect.bottom() - legend_h;
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(lx - 8.0, ly - 6.0), Vec2::new(175.0, legend_h + 6.0)),
                4.0,
                Color32::from_rgba_premultiplied(15, 18, 28, 200),
            );
            for (i, (col, label)) in legend_items.iter().enumerate() {
                let y = ly + i as f32 * 18.0;
                painter.rect_filled(
                    Rect::from_min_size(Pos2::new(lx, y), Vec2::new(12.0, 12.0)),
                    2.0,
                    *col,
                );
                painter.text(
                    Pos2::new(lx + 18.0, y + 6.0),
                    egui::Align2::LEFT_CENTER,
                    *label,
                    egui::FontId::proportional(11.0),
                    Color32::from_gray(200),
                );
            }

            // ---- Camera hint (bottom-left) ----
            let mode_label = if self.show_stl { "STL" } else { "Voxel" };
            painter.text(
                resp.rect.left_bottom() + Vec2::new(10.0, -6.0),
                egui::Align2::LEFT_BOTTOM,
                &format!("Left-drag: orbit   Scroll: zoom   R: reset   Up/Down: prev/next   S: {} view ({})", mode_label, if self.show_stl { "press S for voxel" } else { "press S for STL" }),
                egui::FontId::proportional(10.0),
                Color32::from_gray(70),
            );
        });
    }
}

fn fill_color(pct: f32) -> Color32 {
    if pct < 10.0 {
        Color32::from_rgb(60, 200, 100)
    } else if pct < 40.0 {
        Color32::from_rgb(220, 190, 40)
    } else {
        Color32::from_rgb(210, 80, 50)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: pool-explorer <pool_dir>");
        eprintln!("       pool-explorer <entry.npy>");
        return Ok(());
    }

    let pool_path = PathBuf::from(&args[0]);
    let entries = scan_pool(&pool_path);
    if entries.is_empty() {
        eprintln!(
            "No .npy entries found under '{}'. Run: anysearch extract ...",
            pool_path.display()
        );
        return Ok(());
    }

    let dir_label = pool_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?");
    let title = format!("Pool Explorer — {} ({} entries)", dir_label, entries.len());
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(&title)
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Pool Explorer",
        options,
        Box::new(move |_cc| Ok(Box::new(PoolExplorerApp::new(&pool_path, entries)))),
    )
}
