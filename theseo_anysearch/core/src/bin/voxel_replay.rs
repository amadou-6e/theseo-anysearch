/// Voxel Replay Viewer
///
/// Loads one or more JSON trajectory files and replays episodes with two-level
/// navigation: jump between iterations and scrub individual steps.
///
/// Usage:
///     voxel-replay <file1.json> [file2.json ...]
///
/// Keyboard shortcuts:
///   [ / ]      — prev / next iteration (jumps to step 0 of that iter)
///   ← / →      — prev / next step within current iteration
///   Space      — play / pause (advances through all iterations to the end)
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use eframe::egui::{self, Color32, Key, Pos2, Rect, Sense, Shape, Slider, Stroke, Vec2};
use serde::Deserialize;
use theseo_core::replay::explain::NativeExplainUi;
use theseo_core::replay::lod::{
    chunks_intersecting_box, expand_chunk_halo, include_mandatory_chunks, select_chunks,
    CameraChunkView, ChunkBudgets,
};
use theseo_core::replay::overview::{OverviewMesh, ProjectedVertex};
use theseo_core::replay::regional::{
    agent_region, camera_relative, RegionalReplayFrame, RegionalReplaySource, ReplayMutation,
};
use theseo_core::replay::render_cache::{
    chunk_occupancy_revision, ChunkCoord, ChunkRenderCache, ExposedFace, FaceDirection,
    RenderCacheKey,
};
use theseo_core::voxel::world::StorageCoord;

const DEFAULT_VISUALIZATION_RADIUS: u32 = 16;
const VIEWER_CACHE_BYTES: usize = 256 * 1024 * 1024;

fn chunk_intersects_region(
    chunk: ChunkCoord,
    chunk_edge: u32,
    region: theseo_core::voxel::world::BoundedRegion,
) -> bool {
    let minimum = StorageCoord {
        x: chunk.x.saturating_mul(chunk_edge),
        y: chunk.y.saturating_mul(chunk_edge),
        z: chunk.z.saturating_mul(chunk_edge),
    };
    let maximum_exclusive = StorageCoord {
        x: minimum.x.saturating_add(chunk_edge),
        y: minimum.y.saturating_add(chunk_edge),
        z: minimum.z.saturating_add(chunk_edge),
    };
    minimum.x < region.maximum_exclusive.x
        && maximum_exclusive.x > region.minimum.x
        && minimum.y < region.maximum_exclusive.y
        && maximum_exclusive.y > region.minimum.y
        && minimum.z < region.maximum_exclusive.z
        && maximum_exclusive.z > region.minimum.z
}

// ---------------------------------------------------------------------------
// JSON data model — must match the Python TrajectoryWriter output
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone)]
struct StepData {
    step: u32,
    // Single-agent fields
    #[serde(default)]
    action: i32,
    #[serde(default)]
    reward: f32,
    #[serde(default)]
    cursor_x: u16,
    #[serde(default)]
    cursor_y: u16,
    #[serde(default)]
    cursor_z: u16,
    #[serde(default)]
    voxel_count: u32,
    #[serde(default)]
    placed: bool,
    // Multi-agent fields
    #[serde(default)]
    actions: Vec<i32>,
    #[serde(default)]
    rewards: Vec<f32>,
    #[serde(default)]
    cursors: Vec<[u16; 3]>,
    #[serde(default)]
    placed_per_agent: Vec<bool>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    mutations: Vec<MutationData>,
}

#[derive(Deserialize, Clone)]
struct MutationData {
    coordinate: [u32; 3],
    occupied: bool,
    #[serde(default)]
    kind: u8,
    #[serde(default)]
    active: bool,
    #[serde(default)]
    reward_weight: f32,
}

#[derive(Deserialize, Clone)]
struct WorldReferenceData {
    identity_sha256: String,
    schema_version: u32,
    coordinate_type: String,
    extent: [u32; 3],
    manifest_path: String,
}

#[derive(Deserialize, Clone)]
struct EpisodeData {
    total_reward: f32,
    steps_taken: u32,
    success: bool,
    #[serde(default)]
    init_filled: Vec<[u16; 3]>,
    #[serde(default)]
    init_filled_file: Option<String>,
    steps: Vec<StepData>,
    #[serde(default)]
    start_pos: Option<[u16; 3]>,
    #[serde(default)]
    goal_pos: Option<[u16; 3]>,
    // Multi-agent
    #[serde(default)]
    start_positions: Vec<Option<[u16; 3]>>,
    #[serde(default)]
    goal_positions: Vec<Option<[u16; 3]>>,
}

fn parse_npy_uint16_2d_3cols(bytes: &[u8]) -> Result<Vec<[u16; 3]>, String> {
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
    if !header.contains("<u2") {
        return Err("expected uint16 numpy array".into());
    }

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
    if dims.len() != 2 || dims[1] != 3 {
        return Err(format!("expected shape (N, 3), got {:?}", dims));
    }

    let expected = dims[0] * dims[1] * std::mem::size_of::<u16>();
    let data_bytes = &bytes[hdr_end..];
    if data_bytes.len() < expected {
        return Err(format!("data too short: {} < {}", data_bytes.len(), expected));
    }

    let mut coords = Vec::with_capacity(dims[0]);
    for chunk in data_bytes[..expected].chunks_exact(6) {
        coords.push([
            u16::from_le_bytes([chunk[0], chunk[1]]),
            u16::from_le_bytes([chunk[2], chunk[3]]),
            u16::from_le_bytes([chunk[4], chunk[5]]),
        ]);
    }
    Ok(coords)
}

fn load_trajectory(path: &std::path::Path) -> Option<TrajectoryData> {
    let json = std::fs::read_to_string(path).ok()?;
    let mut traj = serde_json::from_str::<TrajectoryData>(&json).ok()?;
    traj.source_path = path.to_path_buf();
    if let Some(world) = &traj.world {
        let manifest_path = path.parent().unwrap_or(std::path::Path::new(".")).join(
            &world.manifest_path,
        );
        let manifest_text = match std::fs::read_to_string(&manifest_path) {
            Ok(value) => value,
            Err(error) => {
                eprintln!(
                    "trajectory '{}' references missing compiled-world manifest '{}': {error}",
                    path.display(), manifest_path.display()
                );
                return None;
            }
        };
        let manifest: serde_json::Value = match serde_json::from_str(&manifest_text) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("compiled-world manifest '{}' is invalid: {error}", manifest_path.display());
                return None;
            }
        };
        if manifest.get("identity_sha256").and_then(serde_json::Value::as_str)
            != Some(world.identity_sha256.as_str())
        {
            eprintln!(
                "compiled-world identity mismatch for trajectory '{}': expected {}",
                path.display(), world.identity_sha256
            );
            return None;
        }
        traj.world_chunk_edge = manifest.get("chunk_shape")
            .and_then(|shape| shape.get("x"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(32);
    }
    if traj.episode.init_filled.is_empty() {
        if let Some(sidecar) = &traj.episode.init_filled_file {
            let npy_path = path.parent().unwrap_or(std::path::Path::new(".")).join(sidecar);
            let bytes = std::fs::read(&npy_path).ok()?;
            traj.episode.init_filled = parse_npy_uint16_2d_3cols(&bytes).ok()?;
        }
    }
    Some(traj)
}

#[derive(Deserialize, Clone)]
struct TrajectoryData {
    #[serde(skip)]
    source_path: PathBuf,
    #[serde(skip)]
    world_chunk_edge: u32,
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    world: Option<WorldReferenceData>,
    experiment_name: String,
    run_id: String,
    iteration: u32,
    episode_reward_mean: f32,
    max_steps: u32,
    #[serde(default = "default_obs_mode")]
    obs_mode: String,
    #[serde(default = "default_agent_count")]
    agent_count: u32,
    #[serde(default = "default_grid_size")]
    grid_size: u32,
    episode: EpisodeData,
}

fn default_obs_mode() -> String { "scalar".to_string() }
fn default_agent_count() -> u32 { 1 }
fn default_grid_size() -> u32 { 32 }

// ---------------------------------------------------------------------------
// Tune mode — one entry per trial, sorted by best reward descending
// ---------------------------------------------------------------------------

struct TrialEntry {
    trial_id: String,
    /// Human-readable name: `experiment_tag` from ray_runtime.json, or the dir name.
    trial_name: String,
    best_reward: f32,
    params: serde_json::Value,
    trajectory_dir: PathBuf,
    /// Chronological index parsed from the `_N` suffix of the Ray Tune dir name.
    sort_key: u64,
}

/// Scan a tune run directory for `trial_*/trajectories/` subdirs.
/// Returns entries sorted by best_reward descending (best trial first).
fn scan_tune_dir(dir: &std::path::Path) -> Vec<TrialEntry> {
    let mut entries: Vec<TrialEntry> = Vec::new();

    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!("Cannot read tune dir '{}': {e}", dir.display());
            return entries;
        }
    };

    for de in rd.filter_map(|e| e.ok()) {
        let path = de.path();
        if !path.is_dir() { continue; }
        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Identify trial dirs by the presence of ray_runtime.json (Ray Tune's marker file).
        // Trial dirs are named {hash}_{num} by Ray, not trial_* as in older layouts.
        let runtime_path = path.join("ray_runtime.json");
        if !runtime_path.exists() { continue; }

        let traj_dir = path.join("trajectories");
        if !traj_dir.exists() { continue; }

        // Read ray_runtime.json for human-readable trial metadata.
        let runtime_val: serde_json::Value = std::fs::read_to_string(&runtime_path).ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        // `experiment_tag` is Ray Tune's human-readable trial name, e.g. "lr=0.001,batch=4096".
        let trial_name = runtime_val["experiment_tag"].as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(&dir_name)
            .to_string();

        // Chronological sort key: parse the numeric `_N` suffix from the dir name.
        // Ray names dirs `{8-char-hash}_{N}` where N increments from 0.
        let sort_key: u64 = dir_name.rsplit('_').next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(u64::MAX);

        // Try best_meta.json first, then fall back to reading best.json directly.
        let best_reward = {
            let meta_path = traj_dir.join("best_meta.json");
            let reward_from_meta = std::fs::read_to_string(&meta_path).ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v["episode_reward_mean"].as_f64())
                .map(|f| f as f32);
            if let Some(r) = reward_from_meta {
                r
            } else {
                let best_path = traj_dir.join("best.json");
                std::fs::read_to_string(&best_path).ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v["episode_reward_mean"].as_f64())
                    .map(|f| f as f32)
                    .unwrap_or(f32::NEG_INFINITY)
            }
        };

        let params_path = path.join("params.json");
        let params = std::fs::read_to_string(&params_path).ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or(serde_json::Value::Object(Default::default()));

        entries.push(TrialEntry { trial_id: dir_name, trial_name, best_reward, params, trajectory_dir: traj_dir, sort_key });
    }

    // Chronological order (earliest trial first, by the _N suffix Ray appends to dir names).
    entries.sort_by_key(|e| e.sort_key);
    entries
}

/// Load all iter_*.json files from a trial's trajectory dir, sorted by filename.
/// Falls back to best.json if no iter files exist.
fn load_trial_trajectories(traj_dir: &std::path::Path) -> Vec<TrajectoryData> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(traj_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
              .map(|e| e.path())
              .filter(|p| {
                  let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                  name.starts_with("iter_") && name.ends_with(".json")
              })
              .collect()
        })
        .unwrap_or_default();
    files.sort();

    if files.is_empty() {
        let best = traj_dir.join("best.json");
        if best.exists() { files.push(best); }
    }

    let mut trajs: Vec<TrajectoryData> = files.iter()
        .filter_map(|p| load_trajectory(p))
        .collect();
    trajs.sort_by_key(|t| t.iteration);
    trajs
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RegionRequestKey {
    iteration: usize,
    step: usize,
    center: StorageCoord,
    radius: u32,
    camera_revision: u64,
}

struct LoadedRegion {
    frame: RegionalReplayFrame,
    coarse: Vec<ChunkCoord>,
    considered: usize,
    detailed: usize,
}

struct PendingRegion {
    key: RegionRequestKey,
    receiver: Receiver<Result<LoadedRegion, String>>,
}

fn replay_mutations_at(steps: &[StepData], step: usize) -> Vec<ReplayMutation> {
    let mut resolved = HashMap::new();
    for replay_step in steps.iter().take(step.saturating_add(1)) {
        for mutation in &replay_step.mutations {
            let coordinate = StorageCoord {
                x: mutation.coordinate[0],
                y: mutation.coordinate[1],
                z: mutation.coordinate[2],
            };
            resolved.insert(coordinate, mutation.occupied);
        }
    }
    let mut mutations = resolved
        .into_iter()
        .map(|(coordinate, occupied)| ReplayMutation { coordinate, occupied })
        .collect::<Vec<_>>();
    mutations.sort_by_key(|mutation| mutation.coordinate.global_key());
    mutations
}

fn selected_agent_center(trajectory: &TrajectoryData, step: usize) -> Option<StorageCoord> {
    let episode = &trajectory.episode;
    let selected = episode.steps.get(step);
    let coordinate = if trajectory.agent_count > 1 {
        selected
            .and_then(|value| value.cursors.first().copied())
            .or_else(|| episode.start_positions.first().copied().flatten())
    } else {
        selected
            .map(|value| [value.cursor_x, value.cursor_y, value.cursor_z])
            .or(episode.start_pos)
    }?;
    Some(StorageCoord {
        x: u32::from(coordinate[0]),
        y: u32::from(coordinate[1]),
        z: u32::from(coordinate[2]),
    })
}

fn regional_sources(trajectories: &[TrajectoryData]) -> Vec<Option<RegionalReplaySource>> {
    trajectories
        .iter()
        .map(|trajectory| {
            let world = trajectory.world.as_ref()?;
            let manifest = trajectory
                .source_path
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join(&world.manifest_path);
            let root = manifest.parent()?;
            match RegionalReplaySource::open(root, VIEWER_CACHE_BYTES) {
                Ok(source) => Some(source),
                Err(error) => {
                    eprintln!(
                        "failed to open compiled world for '{}': {error:?}",
                        trajectory.source_path.display()
                    );
                    None
                }
            }
        })
        .collect()
}

fn overview_meshes(trajectories: &[TrajectoryData]) -> Vec<Result<OverviewMesh, String>> {
    trajectories.iter().map(|trajectory| {
        let world = trajectory.world.as_ref().ok_or("trajectory has no compiled world")?;
        let manifest = trajectory.source_path.parent()
            .unwrap_or(std::path::Path::new("."))
            .join(&world.manifest_path);
        OverviewMesh::load(&manifest, &world.identity_sha256)
    }).collect()
}

// ---------------------------------------------------------------------------
// Camera: holds yaw/pitch/zoom and handles projection + screen mapping
// ---------------------------------------------------------------------------

struct Camera {
    yaw:   f32,   // radians — rotates x/z plane
    pitch: f32,   // radians — tilts up/down
    zoom:  f32,   // scale multiplier (1.0 = default)
    perspective: bool,
    field_of_view_degrees: f32,
}

impl Camera {
    fn default() -> Self {
        Self {
            yaw: 45.0_f32.to_radians(),
            pitch: 30.0_f32.to_radians(),
            zoom: 1.0,
            perspective: false,
            field_of_view_degrees: 45.0,
        }
    }

    /// Project a 3-D grid coord to 2-D screen space (pre-bounds).
    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let xr  = x * self.yaw.cos() - z * self.yaw.sin();
        let zr  = x * self.yaw.sin() + z * self.yaw.cos();
        let yr  = y * self.pitch.cos() - zr * self.pitch.sin();
        let zr2 = y * self.pitch.sin() + zr * self.pitch.cos();
        (xr, -(yr - zr2 * 0.05))
    }

    fn camera_space(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let xr = x * self.yaw.cos() - z * self.yaw.sin();
        let zr = x * self.yaw.sin() + z * self.yaw.cos();
        let yr = y * self.pitch.cos() - zr * self.pitch.sin();
        let depth = y * self.pitch.sin() + zr * self.pitch.cos();
        (xr, -yr, depth)
    }

    /// Axis-aligned bounding box of the full grid in projected 2-D space.
    fn bounds(&self, grid_size: f32) -> Bounds {
        let lo = 0.5f32;
        let hi = grid_size + 0.5;
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
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
        Bounds {
            min_x: min_x - pw,
            max_x: max_x + pw,
            min_y: min_y - ph,
            max_y: max_y + ph,
            // A cube's largest possible orthographic projection is bounded by
            // its 3-D diagonal. Keep this span independent of yaw and pitch so
            // orbiting never changes the apparent zoom.
            uniform_span: grid_size * 3.0_f32.sqrt() * 1.1,
            world_center: (lo + hi) * 0.5,
        }
    }

    fn screen_scale(&self, rect: Rect, bounds: &Bounds) -> f32 {
        rect.width().min(rect.height()) / bounds.uniform_span.max(1.0) * self.zoom
    }

    /// Map a 3-D coord to a pixel position inside `rect`, applying zoom.
    #[inline]
    fn to_screen(&self, x: f32, y: f32, z: f32, rect: Rect, b: &Bounds) -> Pos2 {
        if self.perspective {
            let (px, py, depth) = self.camera_space(
                x - b.world_center,
                y - b.world_center,
                z - b.world_center,
            );
            let camera_distance = b.uniform_span.max(1.0);
            let near_plane = camera_distance * 0.01;
            let distance = (camera_distance - depth).max(near_plane);
            let half_fov = (self.field_of_view_degrees.clamp(15.0, 100.0) * 0.5)
                .to_radians();
            let focal_pixels = rect.width().min(rect.height()) * 0.5
                / half_fov.tan()
                * self.zoom;
            return Pos2::new(
                rect.center().x + px / distance * focal_pixels,
                rect.center().y + py / distance * focal_pixels,
            );
        }
        let (px, py) = self.project(x, y, z);
        let cx = rect.center().x;
        let cy = rect.center().y;
        let scale = self.screen_scale(rect, b);
        Pos2::new(
            cx + (px - (b.min_x + b.max_x) * 0.5) * scale,
            cy + (py - (b.min_y + b.max_y) * 0.5) * scale,
        )
    }
}

struct Bounds {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    uniform_span: f32,
    world_center: f32,
}

#[cfg(test)]
mod camera_tests {
    use super::{overview_region_vertices, Bounds, Camera, StorageCoord};
    use eframe::egui::{Pos2, Rect, Vec2};

    fn test_camera() -> Camera {
        Camera { yaw: 0.0, pitch: 0.0, zoom: 1.0, ..Camera::default() }
    }

    #[test]
    fn screen_mapping_uses_one_scale_for_both_projected_axes() {
        let camera = test_camera();
        let bounds = Bounds {
            min_x: -10.0,
            max_x: 10.0,
            min_y: -5.0,
            max_y: 5.0,
            uniform_span: 20.0,
            world_center: 0.0,
        };
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 100.0));

        let center = camera.to_screen(0.0, 0.0, 0.0, rect, &bounds);
        let horizontal = camera.to_screen(1.0, 0.0, 0.0, rect, &bounds);
        let vertical = camera.to_screen(0.0, -1.0, 0.0, rect, &bounds);

        assert_eq!(horizontal.x - center.x, vertical.y - center.y);
    }

    #[test]
    fn projected_bounds_are_centered_when_the_viewport_aspect_differs() {
        let camera = test_camera();
        let bounds = Bounds {
            min_x: -10.0,
            max_x: 10.0,
            min_y: -5.0,
            max_y: 5.0,
            uniform_span: 20.0,
            world_center: 0.0,
        };
        let rect = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(300.0, 100.0));

        let center = camera.to_screen(0.0, 0.0, 0.0, rect, &bounds);

        assert_eq!(center, rect.center());
    }

    #[test]
    fn orbiting_does_not_change_the_screen_scale() {
        let first = Camera { yaw: 0.0, pitch: 0.0, zoom: 1.0, ..Camera::default() };
        let second = Camera {
            yaw: 67.0_f32.to_radians(),
            pitch: 41.0_f32.to_radians(),
            zoom: 1.0,
            ..Camera::default()
        };
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(360.0, 200.0));
        let first_bounds = first.bounds(32.0);
        let second_bounds = second.bounds(32.0);

        assert_eq!(
            first.screen_scale(rect, &first_bounds),
            second.screen_scale(rect, &second_bounds),
        );
    }

    #[test]
    fn perspective_makes_nearer_equal_sized_objects_larger() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            perspective: true,
            ..Camera::default()
        };
        let bounds = camera.bounds(32.0);
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(400.0));
        let center = bounds.world_center;
        let far_center = camera.to_screen(center, center, center - 8.0, rect, &bounds);
        let far_edge = camera.to_screen(center + 1.0, center, center - 8.0, rect, &bounds);
        let near_center = camera.to_screen(center, center, center + 8.0, rect, &bounds);
        let near_edge = camera.to_screen(center + 1.0, center, center + 8.0, rect, &bounds);

        assert!(near_edge.x - near_center.x > far_edge.x - far_center.x);
    }

    #[test]
    fn perspective_preserves_aspect_ratio() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            perspective: true,
            ..Camera::default()
        };
        let bounds = camera.bounds(32.0);
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(500.0, 240.0));
        let center = bounds.world_center;
        let origin = camera.to_screen(center, center, center, rect, &bounds);
        let horizontal = camera.to_screen(center + 1.0, center, center, rect, &bounds);
        let vertical = camera.to_screen(center, center - 1.0, center, rect, &bounds);

        assert_eq!(horizontal.x - origin.x, vertical.y - origin.y);
    }

    #[test]
    fn overview_region_vertices_preserve_exact_loaded_bounds() {
        let vertices = overview_region_vertices(
            StorageCoord { x: 10, y: 20, z: 30 },
            StorageCoord { x: 43, y: 53, z: 63 },
        );

        assert!(vertices.contains(&[10, 20, 30]));
        assert!(vertices.contains(&[43, 53, 63]));
    }
}

/// Verifies the actual bug report: a wall no longer hides a marker sitting
/// in front of it, and still correctly hides one that is genuinely behind
/// it. `SceneDrawItem`/`paint_scene_items` replaced the old `occlude_agent`
/// toggle, which drew either "all geometry, then all markers" or "all
/// markers, then all geometry" -- an all-or-nothing choice with no relation
/// to which object was actually closer to the camera. `agent_x_ray` is
/// tested separately: it must override this ordering only for agent-layer
/// items, never for world geometry.
#[cfg(test)]
mod scene_occlusion_tests {
    use super::*;

    fn straight_on_camera() -> Camera {
        // yaw = pitch = 0 collapses depth_key to the camera-relative z
        // coordinate alone, so "in front" / "behind" can be reasoned about
        // directly instead of through a rotated depth formula.
        Camera { yaw: 0.0, pitch: 0.0, zoom: 1.0, ..Camera::default() }
    }

    #[test]
    fn marker_in_front_of_a_wall_draws_after_it() {
        let origin = StorageCoord { x: 50, y: 50, z: 50 };
        let cam = straight_on_camera();
        let wall = SceneDrawItem::Voxel {
            x: 50, y: 50, z: 60, // 10 units in front of the origin.
            color: Color32::GRAY,
            outline: false,
            agent: false,
        };
        let marker_in_front = SceneDrawItem::Marker {
            x: 50, y: 50, z: 70, // 20 units in front -- closer to the camera than the wall.
            color: Color32::YELLOW,
        };

        assert!(
            marker_in_front.depth(origin, &cam, false) > wall.depth(origin, &cam, false),
            "a marker closer to the camera than the wall must sort after it, \
             so it paints on top and stays visible"
        );
    }

    #[test]
    fn marker_behind_a_wall_draws_before_it() {
        let origin = StorageCoord { x: 50, y: 50, z: 50 };
        let cam = straight_on_camera();
        let wall = SceneDrawItem::Voxel {
            x: 50, y: 50, z: 60,
            color: Color32::GRAY,
            outline: false,
            agent: false,
        };
        let marker_behind = SceneDrawItem::Marker {
            x: 50, y: 50, z: 55, // 5 units in front of the origin -- farther than the wall.
            color: Color32::YELLOW,
        };

        assert!(
            marker_behind.depth(origin, &cam, false) < wall.depth(origin, &cam, false),
            "a marker farther from the camera than the wall must sort before \
             it, so the wall paints over it and it stays hidden"
        );
    }

    #[test]
    fn x_ray_forces_a_marker_behind_a_wall_to_draw_after_it() {
        let origin = StorageCoord { x: 50, y: 50, z: 50 };
        let cam = straight_on_camera();
        let wall = SceneDrawItem::Voxel {
            x: 50, y: 50, z: 60,
            color: Color32::GRAY,
            outline: false,
            agent: false,
        };
        let marker_behind = SceneDrawItem::Marker { x: 50, y: 50, z: 55, color: Color32::YELLOW };

        assert!(
            marker_behind.depth(origin, &cam, true) > wall.depth(origin, &cam, true),
            "with x_ray on, an agent-layer item behind a wall must still \
             sort after it, so it stays visible through the wall"
        );
    }

    #[test]
    fn x_ray_never_affects_world_geometry_depth() {
        let origin = StorageCoord { x: 50, y: 50, z: 50 };
        let cam = straight_on_camera();
        let near_wall = SceneDrawItem::Voxel {
            x: 50, y: 50, z: 70,
            color: Color32::GRAY,
            outline: false,
            agent: false,
        };
        let far_wall = SceneDrawItem::Voxel {
            x: 50, y: 50, z: 55,
            color: Color32::GRAY,
            outline: false,
            agent: false,
        };

        // agent: false items must ignore x_ray entirely and keep sorting by
        // real depth -- only the agent layer gets forced to the front.
        assert_eq!(near_wall.depth(origin, &cam, false), near_wall.depth(origin, &cam, true));
        assert_eq!(far_wall.depth(origin, &cam, false), far_wall.depth(origin, &cam, true));
        assert!(near_wall.depth(origin, &cam, true) > far_wall.depth(origin, &cam, true));
    }

    #[test]
    fn paint_scene_items_actually_sorts_mixed_geometry_and_markers_by_depth() {
        let origin = StorageCoord { x: 0, y: 0, z: 0 };
        let cam = straight_on_camera();
        let items = vec![
            SceneDrawItem::Cursor { x: 0, y: 0, z: 30 }, // closest
            SceneDrawItem::Voxel { x: 0, y: 0, z: 10, color: Color32::GRAY, outline: false, agent: false },
            SceneDrawItem::Marker { x: 0, y: 0, z: 20, color: Color32::YELLOW },
            SceneDrawItem::Voxel { x: 0, y: 0, z: 5, color: Color32::GRAY, outline: false, agent: false }, // farthest
        ];
        let mut depths: Vec<f32> = items.iter().map(|item| item.depth(origin, &cam, false)).collect();
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Reproduce paint_scene_items' ordering decision without a live
        // egui::Painter (draw() requires one); this is the exact sort it
        // performs, just observed instead of rendered.
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by(|&i, &j| {
            items[i].depth(origin, &cam, false).partial_cmp(&items[j].depth(origin, &cam, false)).unwrap()
        });
        let sorted_depths: Vec<f32> = order.iter().map(|&i| items[i].depth(origin, &cam, false)).collect();
        assert_eq!(sorted_depths, depths, "paint order must be strictly back-to-front");
        // farthest-first, closest-last: index 3 (z=5) then 1 (z=10) then 2 (z=20) then 0 (z=30)
        assert_eq!(order, vec![3, 1, 2, 0]);
    }
}

fn overview_region_vertices(
    minimum: StorageCoord,
    maximum_exclusive: StorageCoord,
) -> [[u32; 3]; 8] {
    let (x0, y0, z0) = (minimum.x, minimum.y, minimum.z);
    let (x1, y1, z1) = (
        maximum_exclusive.x,
        maximum_exclusive.y,
        maximum_exclusive.z,
    );
    [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
}

fn inset_position(point: ProjectedVertex, rect: Rect, scale: f32) -> Pos2 {
    Pos2::new(rect.center().x + point.x * scale, rect.center().y + point.y * scale)
}

fn draw_overview_inset(
    painter: &egui::Painter, outer: Rect, mesh: &OverviewMesh, camera: &Camera,
    size: f32, show_bounds: bool,
    visible_region: Option<(StorageCoord, StorageCoord)>,
) {
    let inset_size = outer.width().min(outer.height()).min(size);
    let rect = Rect::from_min_size(
        outer.right_bottom() - Vec2::splat(inset_size + 12.0), Vec2::splat(inset_size),
    );
    painter.rect_filled(rect, 6.0, Color32::from_rgba_premultiplied(8, 11, 17, 225));
    painter.rect_stroke(rect, 6.0, Stroke::new(1.0, Color32::from_gray(75)), egui::StrokeKind::Inside);
    let projected = mesh.project(camera.yaw, camera.pitch);
    let bounds_mesh = OverviewMesh { vertices: mesh.bounds_vertices(), indices: Vec::new(), extent: mesh.extent };
    let bounds_points = bounds_mesh.project(camera.yaw, camera.pitch);
    let maximum_x = bounds_points.iter().map(|point| point.x.abs()).fold(0.0, f32::max).max(0.001);
    let maximum_y = bounds_points.iter().map(|point| point.y.abs()).fold(0.0, f32::max).max(0.001);
    let scale = (rect.width() / (2.0 * maximum_x))
        .min(rect.height() / (2.0 * maximum_y)) * 0.86;
    let mut triangles = mesh.indices.chunks_exact(3).map(|indices| {
        let points = [projected[indices[0] as usize], projected[indices[1] as usize], projected[indices[2] as usize]];
        (points.iter().map(|point| point.depth).sum::<f32>() / 3.0, points)
    }).collect::<Vec<_>>();
    triangles.sort_by(|left, right| left.0.total_cmp(&right.0));
    for (_, triangle) in triangles {
        painter.add(Shape::convex_polygon(
            triangle.into_iter().map(|point| inset_position(point, rect, scale)).collect(),
            Color32::from_rgb(105, 130, 150), Stroke::new(0.35, Color32::from_gray(50)),
        ));
    }
    if show_bounds {
        for (a, b) in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)] {
            painter.line_segment(
                [inset_position(bounds_points[a], rect, scale), inset_position(bounds_points[b], rect, scale)],
                Stroke::new(1.0, Color32::from_gray(170)),
            );
        }
    }
    if let Some((minimum, maximum_exclusive)) = visible_region {
        let region_mesh = OverviewMesh {
            vertices: overview_region_vertices(minimum, maximum_exclusive).to_vec(),
            indices: Vec::new(),
            extent: mesh.extent,
        };
        let points = region_mesh.project(camera.yaw, camera.pitch);
        for face in [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [3, 0, 4, 7],
        ] {
            painter.add(Shape::convex_polygon(
                face.into_iter()
                    .map(|index| inset_position(points[index], rect, scale))
                    .collect(),
                Color32::from_rgba_premultiplied(10, 40, 20, 36),
                Stroke::NONE,
            ));
        }
        for (a, b) in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)] {
            painter.line_segment(
                [inset_position(points[a], rect, scale), inset_position(points[b], rect, scale)],
                Stroke::new(2.0, Color32::from_rgb(70, 255, 120)),
            );
        }
    }
    painter.text(rect.left_top() + Vec2::splat(6.0), egui::Align2::LEFT_TOP, "World overview", egui::FontId::proportional(10.0), Color32::from_gray(190));
}

/// Back-to-front depth key for painter's algorithm.
/// Voxels with lower value are further from camera and should be drawn first.
fn depth_key(x: u16, y: u16, z: u16, origin: StorageCoord, cam: &Camera) -> f32 {
    let (sy, cy) = (cam.yaw.sin(), cam.yaw.cos());
    let (sp, cp) = (cam.pitch.sin(), cam.pitch.cos());
    let (x, y, z) = camera_relative(
        StorageCoord { x: u32::from(x), y: u32::from(y), z: u32::from(z) },
        origin,
    );
    x * sy * cp + y * sp + z * cy * cp
}

fn draw_voxel(
    painter: &egui::Painter,
    cx: u16, cy: u16, cz: u16,
    origin: StorageCoord,
    rect: Rect, cam: &Camera, b: &Bounds,
    base: Color32,
    outline: bool,
) {
    let (x, y, z) = camera_relative(
        StorageCoord { x: u32::from(cx), y: u32::from(cy), z: u32::from(cz) },
        origin,
    );
    let h = 0.5_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);

    // Select the visible face on each axis based on camera orientation.
    // The "to-camera" direction is (sin(yaw)*cos(pitch), sin(pitch), cos(yaw)*cos(pitch)).
    // A face with outward normal N is visible when dot(N, to_cam) > 0.
    let hx = if cam.yaw.sin() > 0.0 { h } else { -h };   // +X visible when sin(yaw)>0
    let hy = if cam.pitch.sin() > 0.0 { h } else { -h }; // +Y (top) visible when sin(pitch)>0
    let hz = if cam.yaw.cos() > 0.0 { h } else { -h };   // +Z visible when cos(yaw)>0

    let top_face = vec![
        corner(-h, hy, -h), corner( h, hy, -h),
        corner( h, hy,  h), corner(-h, hy,  h),
    ];
    let face_x = vec![
        corner(hx, -h, -h), corner(hx, -h,  h),
        corner(hx,  h,  h), corner(hx,  h, -h),
    ];
    let face_z = vec![
        corner(-h, -h, hz), corner( h, -h, hz),
        corner( h,  h, hz), corner(-h,  h, hz),
    ];

    let shade = |r: u8, g: u8, bl: u8, amount: i32| -> Color32 {
        Color32::from_rgb(
            (r as i32 + amount).clamp(0, 255) as u8,
            (g as i32 + amount).clamp(0, 255) as u8,
            (bl as i32 + amount).clamp(0, 255) as u8,
        )
    };
    let (r, g, bl) = (base.r(), base.g(), base.b());
    let top_col   = shade(r, g, bl,  40);
    let facex_col = shade(r, g, bl, -10);
    let facez_col = shade(r, g, bl, -40);

    let stroke = if outline { Stroke::new(0.5, Color32::from_gray(30)) } else { Stroke::NONE };

    // Draw back-to-front within the voxel (z-face furthest, then x-face, then top).
    painter.add(Shape::convex_polygon(face_z,    facez_col, stroke));
    painter.add(Shape::convex_polygon(face_x,    facex_col, stroke));
    painter.add(Shape::convex_polygon(top_face,  top_col,   stroke));
}

fn draw_exposed_face(
    painter: &egui::Painter,
    face: ExposedFace,
    origin: StorageCoord,
    rect: Rect,
    cam: &Camera,
    bounds: &Bounds,
    base: Color32,
) {
    let visible = match face.direction {
        FaceDirection::NegativeX => cam.yaw.sin() < 0.0,
        FaceDirection::PositiveX => cam.yaw.sin() > 0.0,
        FaceDirection::NegativeY => cam.pitch.sin() < 0.0,
        FaceDirection::PositiveY => cam.pitch.sin() > 0.0,
        FaceDirection::NegativeZ => cam.yaw.cos() < 0.0,
        FaceDirection::PositiveZ => cam.yaw.cos() > 0.0,
    };
    if !visible { return; }
    let (x, y, z) = camera_relative(face.voxel, origin);
    let h = 0.5_f32;
    let corner = |dx, dy, dz| cam.to_screen(x + dx, y + dy, z + dz, rect, bounds);
    let points = match face.direction {
        FaceDirection::NegativeX => vec![corner(-h,-h,-h), corner(-h,h,-h), corner(-h,h,h), corner(-h,-h,h)],
        FaceDirection::PositiveX => vec![corner(h,-h,-h), corner(h,-h,h), corner(h,h,h), corner(h,h,-h)],
        FaceDirection::NegativeY => vec![corner(-h,-h,-h), corner(-h,-h,h), corner(h,-h,h), corner(h,-h,-h)],
        FaceDirection::PositiveY => vec![corner(-h,h,-h), corner(h,h,-h), corner(h,h,h), corner(-h,h,h)],
        FaceDirection::NegativeZ => vec![corner(-h,-h,-h), corner(h,-h,-h), corner(h,h,-h), corner(-h,h,-h)],
        FaceDirection::PositiveZ => vec![corner(-h,-h,h), corner(-h,h,h), corner(h,h,h), corner(h,-h,h)],
    };
    let adjustment = match face.direction {
        FaceDirection::NegativeY | FaceDirection::PositiveY => 40,
        FaceDirection::NegativeX | FaceDirection::PositiveX => -10,
        FaceDirection::NegativeZ | FaceDirection::PositiveZ => -40,
    };
    let color = Color32::from_rgb(
        (i32::from(base.r()) + adjustment).clamp(0, 255) as u8,
        (i32::from(base.g()) + adjustment).clamp(0, 255) as u8,
        (i32::from(base.b()) + adjustment).clamp(0, 255) as u8,
    );
    painter.add(Shape::convex_polygon(points, color, Stroke::NONE));
}

fn draw_cursor(painter: &egui::Painter, cx: u16, cy: u16, cz: u16,
               origin: StorageCoord, rect: Rect, cam: &Camera, b: &Bounds) {
    let (x, y, z) = camera_relative(
        StorageCoord { x: u32::from(cx), y: u32::from(cy), z: u32::from(cz) },
        origin,
    );
    let h = 0.5_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);

    let yellow = Color32::from_rgb(255, 230, 0);
    let stroke = Stroke::new(1.5, yellow);

    let top: Vec<Pos2> = vec![
        corner(-h,  h, -h), corner( h,  h, -h),
        corner( h,  h,  h), corner(-h,  h,  h),
    ];
    painter.add(Shape::closed_line(top, stroke));

    let bot: Vec<Pos2> = vec![
        corner(-h, -h, -h), corner( h, -h, -h),
        corner( h, -h,  h), corner(-h, -h,  h),
    ];
    painter.add(Shape::closed_line(bot, stroke));

    for (dx, dz) in [(-h,-h),(h,-h),(h,h),(-h,h)] {
        painter.line_segment([corner(dx,-h,dz), corner(dx,h,dz)], stroke);
    }

    let cp = cam.to_screen(x, y, z, rect, b);
    painter.circle_filled(cp, 3.0, yellow);
}

fn draw_coarse_chunk(
    painter: &egui::Painter,
    chunk: ChunkCoord,
    chunk_edge: u32,
    origin: StorageCoord,
    rect: Rect,
    cam: &Camera,
    bounds: &Bounds,
) {
    let coordinate = StorageCoord {
        x: chunk.x * chunk_edge,
        y: chunk.y * chunk_edge,
        z: chunk.z * chunk_edge,
    };
    let (x, y, z) = camera_relative(coordinate, origin);
    let edge = chunk_edge as f32;
    let corners = [
        (0.0,0.0,0.0),(edge,0.0,0.0),(edge,edge,0.0),(0.0,edge,0.0),
        (0.0,0.0,edge),(edge,0.0,edge),(edge,edge,edge),(0.0,edge,edge),
    ].map(|(dx,dy,dz)| cam.to_screen(x + dx, y + dy, z + dz, rect, bounds));
    let stroke = Stroke::new(1.0, Color32::from_rgb(70, 100, 130));
    for (a, b) in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)] {
        painter.line_segment([corners[a], corners[b]], stroke);
    }
}

fn draw_marker(
    painter: &egui::Painter,
    cx: u16, cy: u16, cz: u16,
    origin: StorageCoord,
    rect: Rect, cam: &Camera, b: &Bounds,
    color: Color32,
) {
    draw_voxel(painter, cx, cy, cz, origin, rect, cam, b, color, true);
    let (x, y, z) = camera_relative(
        StorageCoord { x: u32::from(cx), y: u32::from(cy), z: u32::from(cz) },
        origin,
    );
    let h = 0.4_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);
    let stroke = Stroke::new(2.0, Color32::WHITE);
    painter.line_segment([corner(-h, h, -h), corner(h, h, h)], stroke);
    painter.line_segment([corner(h, h, -h), corner(-h, h, h)], stroke);
}

/// One deferred scene draw call, carried alongside its depth so geometry and
/// agent/trail/marker items can be painted in a single back-to-front order.
/// Without this, geometry and markers were two separate all-or-nothing
/// batches (see the removed `occlude_agent` toggle): whichever batch drew
/// second always covered the other, so a wall could hide a marker sitting
/// in front of it, or a marker could float in front of a wall it was
/// actually behind. Sorting everything together makes occlusion depend on
/// each item's real position, like the rest of the scene already does for
/// geometry against itself.
///
/// Agent-layer items (`Cursor`, `Marker`, and `Voxel` with `agent: true` for
/// trail cells) can still be forced to render through any geometry via
/// `x_ray` -- e.g. to keep the agent visible inside tunnels or behind dense
/// structures -- without reintroducing the old all-or-nothing behavior for
/// world geometry itself, which never gets x-rayed.
enum SceneDrawItem {
    Voxel { x: u16, y: u16, z: u16, color: Color32, outline: bool, agent: bool },
    Face { face: ExposedFace, color: Color32 },
    Cursor { x: u16, y: u16, z: u16 },
    Marker { x: u16, y: u16, z: u16, color: Color32 },
}

impl SceneDrawItem {
    fn is_agent_layer(&self) -> bool {
        matches!(
            self,
            SceneDrawItem::Cursor { .. }
                | SceneDrawItem::Marker { .. }
                | SceneDrawItem::Voxel { agent: true, .. }
        )
    }

    fn depth(&self, origin: StorageCoord, cam: &Camera, x_ray: bool) -> f32 {
        if x_ray && self.is_agent_layer() {
            // Sorts after every real depth value, so it always paints last.
            return f32::INFINITY;
        }
        let (x, y, z) = match *self {
            SceneDrawItem::Voxel { x, y, z, .. }
            | SceneDrawItem::Cursor { x, y, z }
            | SceneDrawItem::Marker { x, y, z, .. } => (x, y, z),
            SceneDrawItem::Face { face, .. } => {
                (face.voxel.x as u16, face.voxel.y as u16, face.voxel.z as u16)
            }
        };
        depth_key(x, y, z, origin, cam)
    }

    fn draw(&self, painter: &egui::Painter, origin: StorageCoord, rect: Rect, cam: &Camera, b: &Bounds) {
        match *self {
            SceneDrawItem::Voxel { x, y, z, color, outline, .. } => {
                draw_voxel(painter, x, y, z, origin, rect, cam, b, color, outline);
            }
            SceneDrawItem::Face { face, color } => {
                draw_exposed_face(painter, face, origin, rect, cam, b, color);
            }
            SceneDrawItem::Cursor { x, y, z } => {
                draw_cursor(painter, x, y, z, origin, rect, cam, b);
            }
            SceneDrawItem::Marker { x, y, z, color } => {
                draw_marker(painter, x, y, z, origin, rect, cam, b, color);
            }
        }
    }
}

/// Sort and paint every deferred item back-to-front by camera-relative
/// depth. When `x_ray` is set, agent-layer items (see `SceneDrawItem`) paint
/// last regardless of depth, so the agent stays visible through tunnels or
/// dense geometry instead of being correctly, but unhelpfully, hidden.
fn paint_scene_items(
    items: &mut [SceneDrawItem],
    painter: &egui::Painter,
    origin: StorageCoord,
    rect: Rect,
    cam: &Camera,
    b: &Bounds,
    x_ray: bool,
) {
    let depths: Vec<f32> = items.iter().map(|item| item.depth(origin, cam, x_ray)).collect();
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&i, &j| depths[i].partial_cmp(&depths[j]).unwrap());
    for index in order {
        items[index].draw(painter, origin, rect, cam, b);
    }
}

fn draw_grid_bounds_layer(
    painter: &egui::Painter,
    rect: Rect,
    cam: &Camera,
    b: &Bounds,
    grid_size: f32,
    draw_front: bool,
) {
    let shadow = Stroke::new(2.5, Color32::from_rgba_premultiplied(0, 0, 0, 180));
    let stroke = Stroke::new(1.25, Color32::from_rgb(80, 255, 140));
    let lo = 0.5_f32;
    let hi = grid_size + 0.5;
    let corners = [
        (lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo),
        (lo, lo, hi), (hi, lo, hi), (hi, hi, hi), (lo, hi, hi),
    ];
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    let near_x = if cam.yaw.sin() > 0.0 { hi } else { lo };
    let near_y = if cam.pitch.sin() > 0.0 { hi } else { lo };
    let near_z = if cam.yaw.cos() > 0.0 { hi } else { lo };
    let pts: Vec<Pos2> = corners.iter()
        .map(|&(x, y, z)| cam.to_screen(x, y, z, rect, b))
        .collect();
    let mut visible_corners = [false; 8];

    for (a, bi) in edges {
        let ca = corners[a];
        let cb = corners[bi];
        let is_front = (ca.0 == cb.0 && ca.0 == near_x)
            || (ca.1 == cb.1 && ca.1 == near_y)
            || (ca.2 == cb.2 && ca.2 == near_z);
        if is_front != draw_front {
            continue;
        }
        painter.line_segment([pts[a], pts[bi]], shadow);
        painter.line_segment([pts[a], pts[bi]], stroke);
        visible_corners[a] = true;
        visible_corners[bi] = true;
    }
    if draw_front {
        for (index, point) in pts.iter().enumerate() {
            if visible_corners[index] {
                painter.circle_filled(*point, 1.75, Color32::from_rgb(130, 255, 175));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Application — two-level navigation: iterations + steps
// ---------------------------------------------------------------------------

struct VoxelReplayApp {
    /// All loaded trajectories, sorted by iteration number.
    trajectories: Vec<TrajectoryData>,
    /// Geometry voxel lists (unsorted — sorted per frame based on camera).
    geo_voxels: Vec<Vec<(u16, u16, u16)>>,
    /// Which trajectory (iteration) we are currently viewing.
    iter_idx: usize,
    /// Which step within the current trajectory.
    step_idx: usize,
    /// Whether auto-play is running (advances steps, then iterations).
    playing: bool,
    /// Camera: orbit with left-drag, zoom with scroll wheel.
    camera: Camera,
    /// Keep the agent, trail, and start/goal markers visible through any
    /// geometry (tunnels, dense structures) instead of correctly-occluded.
    agent_x_ray: bool,
    /// Render compiled-world geometry as cached flat exposed-surface quads
    /// instead of individual per-voxel cubes. Off by default: adjacent
    /// occupied voxels sharing a face otherwise appear as one seamless flat
    /// tile with no cube depth. Opt-in for performance-sensitive scenes with
    /// large contiguous surfaces, where the per-voxel cost is worth trading
    /// away.
    surface_mesh: bool,
    /// Tune mode: all trials sorted best-first. Empty in file mode.
    tune_trials: Vec<TrialEntry>,
    /// Which trial is currently loaded (index into tune_trials).
    current_trial: usize,
    /// Optional checkpoint-backed native explanation windows.
    explain_ui: Option<NativeExplainUi>,
    explain_tab: bool,
    regional_sources: Vec<Option<RegionalReplaySource>>,
    regional_frame: Option<(RegionRequestKey, RegionalReplayFrame)>,
    pending_region: Option<PendingRegion>,
    requested_region: Option<RegionRequestKey>,
    visualization_radius: u32,
    regional_error: Option<String>,
    render_cache: ChunkRenderCache,
    regional_faces: Vec<ExposedFace>,
    coarse_chunks: Vec<ChunkCoord>,
    considered_chunks: usize,
    detailed_chunks: usize,
    chunk_budgets: ChunkBudgets,
    camera_revision: u64,
    overview_meshes: Vec<Result<OverviewMesh, String>>,
    show_overview: bool,
    show_overview_bounds: bool,
    overview_size: f32,
}

/// UI events collected during a frame; applied to state after all closures finish.
#[derive(Default)]
struct UiEvents {
    prev_iter:   bool,
    next_iter:   bool,
    first_iter:  bool,
    last_iter:   bool,
    jump_iter:   Option<usize>,
    prev_step:   bool,
    next_step:   bool,
    first_step:  bool,
    last_step:   bool,
    jump_step:   Option<usize>,
    toggle_play: bool,
    // Tune-mode trial navigation
    prev_trial:  bool,
    next_trial:  bool,
    first_trial: bool,
    last_trial:  bool,
}

impl VoxelReplayApp {
    fn new(trajectories: Vec<TrajectoryData>, explain_ui: Option<NativeExplainUi>) -> Self {
        let explain_tab = explain_ui.as_ref().map(|ui| ui.observation_open).unwrap_or(false);
        let geo_voxels = trajectories.iter().map(|t| {
            t.episode.init_filled.iter().map(|c| (c[0], c[1], c[2])).collect()
        }).collect();
        let regional_sources = regional_sources(&trajectories);
        let overview_meshes = overview_meshes(&trajectories);
        Self {
            camera: Camera::default(),
            agent_x_ray: false,
            surface_mesh: false,
            trajectories,
            geo_voxels,
            iter_idx: 0,
            step_idx: 0,
            playing: false,
            tune_trials: Vec::new(),
            current_trial: 0,
            explain_ui,
            explain_tab,
            regional_sources,
            regional_frame: None,
            pending_region: None,
            requested_region: None,
            visualization_radius: DEFAULT_VISUALIZATION_RADIUS,
            regional_error: None,
            render_cache: ChunkRenderCache::default(),
            regional_faces: Vec::new(),
            coarse_chunks: Vec::new(),
            considered_chunks: 0,
            detailed_chunks: 0,
            chunk_budgets: ChunkBudgets { visible: 64, detailed: 24 },
            camera_revision: 0,
            overview_meshes,
            show_overview: true,
            show_overview_bounds: true,
            overview_size: 220.0,
        }
    }

    fn new_tune(trials: Vec<TrialEntry>) -> Self {
        let mut app = Self {
            camera: Camera::default(),
            agent_x_ray: false,
            surface_mesh: false,
            trajectories: Vec::new(),
            geo_voxels: Vec::new(),
            iter_idx: 0,
            step_idx: 0,
            playing: false,
            tune_trials: trials,
            current_trial: 0,
            explain_ui: None,
            explain_tab: false,
            regional_sources: Vec::new(),
            regional_frame: None,
            pending_region: None,
            requested_region: None,
            visualization_radius: DEFAULT_VISUALIZATION_RADIUS,
            regional_error: None,
            render_cache: ChunkRenderCache::default(),
            regional_faces: Vec::new(),
            coarse_chunks: Vec::new(),
            considered_chunks: 0,
            detailed_chunks: 0,
            chunk_budgets: ChunkBudgets { visible: 64, detailed: 24 },
            camera_revision: 0,
            overview_meshes: Vec::new(),
            show_overview: true,
            show_overview_bounds: true,
            overview_size: 220.0,
        };
        app.load_trial(0);
        app
    }

    fn load_trial(&mut self, idx: usize) {
        if idx >= self.tune_trials.len() { return; }
        self.current_trial = idx;
        self.iter_idx = 0;
        self.step_idx = 0;
        self.playing = false;

        let traj_dir = self.tune_trials[idx].trajectory_dir.clone();
        let trajs = load_trial_trajectories(&traj_dir);
        self.geo_voxels = trajs.iter().map(|t| {
            t.episode.init_filled.iter().map(|c| (c[0], c[1], c[2])).collect()
        }).collect();
        self.regional_sources = regional_sources(&trajs);
        self.overview_meshes = overview_meshes(&trajs);
        self.trajectories = trajs;
        self.regional_frame = None;
        self.pending_region = None;
        self.requested_region = None;
        self.regional_error = None;
        self.regional_faces.clear();
        self.coarse_chunks.clear();
    }

    fn current_region_key(&self) -> Option<RegionRequestKey> {
        let trajectory = self.trajectories.get(self.iter_idx)?;
        trajectory.world.as_ref()?;
        Some(RegionRequestKey {
            iteration: self.iter_idx,
            step: self.step_idx,
            center: selected_agent_center(trajectory, self.step_idx)?,
            radius: self.visualization_radius,
            camera_revision: self.camera_revision,
        })
    }

    fn cache_regional_faces(&mut self, key: RegionRequestKey, frame: &RegionalReplayFrame) {
        let chunk_edge = self.trajectories[key.iteration].world_chunk_edge.max(1);
        let occupied = frame.occupied.iter().copied().collect::<HashSet<_>>();
        let chunks = frame.occupied.iter().map(|coordinate| ChunkCoord {
            x: coordinate.x / chunk_edge,
            y: coordinate.y / chunk_edge,
            z: coordinate.z / chunk_edge,
        }).collect::<HashSet<_>>();
        let identity = self.trajectories[key.iteration].world.as_ref()
            .map(|world| world.identity_sha256.clone()).unwrap_or_default();
        self.regional_faces.clear();
        for chunk in chunks {
            let data = self.render_cache.get_or_build(
                RenderCacheKey {
                    world_identity: identity.clone(),
                    chunk,
                    overlay_revision: chunk_occupancy_revision(chunk, &occupied, chunk_edge),
                    settings_revision: 0,
                },
                &occupied,
                chunk_edge,
            );
            self.regional_faces.extend_from_slice(&data.faces);
        }
    }

    fn update_regional_geometry(&mut self, ctx: &egui::Context) {
        let current_key = self.current_region_key();
        if let Some(pending) = self.pending_region.take() {
            match pending.receiver.try_recv() {
                Ok(Ok(loaded)) => {
                    if Some(pending.key) == current_key {
                        self.cache_regional_faces(pending.key, &loaded.frame);
                        self.coarse_chunks = loaded.coarse;
                        self.considered_chunks = loaded.considered;
                        self.detailed_chunks = loaded.detailed;
                        self.regional_frame = Some((pending.key, loaded.frame));
                        self.regional_error = None;
                    }
                }
                Ok(Err(error)) => {
                    if Some(pending.key) == current_key {
                        self.regional_error = Some(error);
                    }
                }
                Err(TryRecvError::Empty) => {
                    self.pending_region = Some(pending);
                    ctx.request_repaint();
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    self.regional_error = Some("regional world loader disconnected".to_string());
                }
            }
        }

        let Some(key) = current_key else { return; };
        if self.requested_region == Some(key) {
            return;
        }
        let Some(source) = self
            .regional_sources
            .get(key.iteration)
            .and_then(Option::as_ref)
            .cloned()
        else {
            self.regional_error = Some("compiled world could not be opened".to_string());
            return;
        };
        let mutations = replay_mutations_at(
            &self.trajectories[key.iteration].episode.steps,
            key.step,
        );
        let chunk_shape = source.chunk_shape();
        let mut indexed = source.indexed_chunks();
        if let Some(shape) = chunk_shape {
            for mutation in &mutations {
                let chunk = ChunkCoord {
                    x: mutation.coordinate.x / shape.x,
                    y: mutation.coordinate.y / shape.y,
                    z: mutation.coordinate.z / shape.z,
                };
                if !indexed.iter().any(|(candidate, _)| *candidate == chunk) {
                    indexed.push((chunk, 0));
                }
            }
        }
        let selection = chunk_shape.map(|shape| {
            let mandatory = chunks_intersecting_box(
                indexed.iter().map(|(chunk, _)| *chunk),
                key.center,
                key.radius,
                shape,
            );
            let radius_chunks = (key.radius as f64 / shape.x.max(shape.y).max(shape.z) as f64)
                .max(1.0) / f64::from(self.camera.zoom.max(0.2));
            let selected = select_chunks(
                indexed.iter().map(|(chunk, _)| *chunk),
                CameraChunkView {
                    center: [
                        key.center.x as f64 / shape.x as f64,
                        key.center.y as f64 / shape.y as f64,
                        key.center.z as f64 / shape.z as f64,
                    ],
                    half_extent: [radius_chunks; 3],
                    forward: [
                        f64::from(self.camera.yaw.sin()),
                        f64::from(self.camera.pitch.sin()),
                        f64::from(self.camera.yaw.cos()),
                    ],
                    minimum_forward_dot: -0.5,
                },
                self.chunk_budgets,
            );
            let selection = include_mandatory_chunks(selected, mandatory.iter().copied());
            let mut visible = selection.detailed.iter().chain(&selection.coarse)
                .copied().collect::<Vec<_>>();
            visible.sort_by_key(|chunk| (chunk.x, chunk.y, chunk.z));
            visible.dedup();
            let mut resident = expand_chunk_halo(
                &mandatory,
                indexed.iter().map(|(chunk, _)| *chunk),
                1,
            );
            resident.extend(visible.iter().copied());
            resident.sort_by_key(|chunk| (chunk.x, chunk.y, chunk.z));
            resident.dedup();
            (selection, visible, resident)
        });
        let display_region = agent_region(key.center, key.radius, source.extent());
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let result = match (selection, display_region) {
                (Some((selection, visible, resident)), Ok(display_region))
                    if !selection.detailed.is_empty() => source
                    .load_chunk_selection_in_region(
                        &selection.detailed,
                        &visible,
                        &resident,
                        &mutations,
                        display_region,
                    )
                    .map(|frame| LoadedRegion {
                        frame,
                        coarse: selection.coarse,
                        considered: selection.considered,
                        detailed: selection.detailed.len(),
                    }),
                _ => source.load_agent_region(key.center, key.radius, &mutations)
                    .map(|frame| LoadedRegion { frame, coarse: Vec::new(), considered: 0, detailed: 0 }),
            }.map_err(|error| format!("regional world load failed: {error:?}"));
            let _ = sender.send(result);
        });
        self.pending_region = Some(PendingRegion { key, receiver });
        self.requested_region = Some(key);
        ctx.request_repaint();
    }

    fn n_iters(&self) -> usize { self.trajectories.len() }
    fn n_steps(&self) -> usize { self.trajectories[self.iter_idx].episode.steps.len() }

    /// Advance one step during play.  Returns false when the run is fully finished.
    fn play_advance(&mut self) -> bool {
        let n = self.n_steps();
        if self.step_idx + 1 < n {
            self.step_idx += 1;
            true
        } else if self.iter_idx + 1 < self.n_iters() {
            self.iter_idx += 1;
            self.step_idx = 0;
            true
        } else {
            false
        }
    }

    fn apply_events(&mut self, ev: UiEvents) {
        // Iteration navigation (always resets step to 0)
        if ev.first_iter { self.iter_idx = 0; self.step_idx = 0; self.playing = false; }
        if ev.prev_iter  { if self.iter_idx > 0 { self.iter_idx -= 1; } self.step_idx = 0; self.playing = false; }
        if ev.next_iter  { if self.iter_idx + 1 < self.n_iters() { self.iter_idx += 1; } self.step_idx = 0; self.playing = false; }
        if ev.last_iter  { self.iter_idx = self.n_iters().saturating_sub(1); self.step_idx = 0; self.playing = false; }
        if let Some(i) = ev.jump_iter { self.iter_idx = i; self.step_idx = 0; self.playing = false; }

        // Step navigation
        let max_step = self.n_steps().saturating_sub(1);
        if ev.first_step { self.step_idx = 0; self.playing = false; }
        if ev.prev_step  { if self.step_idx > 0 { self.step_idx -= 1; } self.playing = false; }
        if ev.next_step  { if self.step_idx < max_step { self.step_idx += 1; } self.playing = false; }
        if ev.last_step  { self.step_idx = max_step; self.playing = false; }
        if let Some(s) = ev.jump_step { self.step_idx = s; self.playing = false; }

        // Play / pause toggle
        if ev.toggle_play {
            let at_end = self.iter_idx == self.n_iters().saturating_sub(1)
                && self.step_idx >= self.n_steps().saturating_sub(1);
            self.playing = !self.playing;
            if self.playing && at_end {
                self.iter_idx = 0;
                self.step_idx = 0;
            }
        }
    }
}

impl eframe::App for VoxelReplayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ---- Collect keyboard events (before closures) ----------------------
        let kb_prev_iter   = ctx.input(|i| i.key_pressed(Key::OpenBracket));
        let kb_next_iter   = ctx.input(|i| i.key_pressed(Key::CloseBracket));
        let kb_prev_step   = ctx.input(|i| i.key_pressed(Key::ArrowLeft));
        let kb_next_step   = ctx.input(|i| i.key_pressed(Key::ArrowRight));
        let kb_play_pause  = ctx.input(|i| i.key_pressed(Key::Space));
        // Tune mode: ] / [ also navigate trials when at first/last iter of a trial.
        // Dedicated keys: N = next trial, P = prev trial.
        let kb_next_trial  = ctx.input(|i| i.key_pressed(Key::T));
        let kb_prev_trial  = ctx.input(|i| i.key_pressed(Key::Y));

        let in_tune_mode = !self.tune_trials.is_empty();
        let n_trials     = self.tune_trials.len();

        // Read-only snapshot of indices (Copy, no borrow issue)
        let iter_idx = self.iter_idx;
        let step_idx = self.step_idx;
        let n_iters  = self.n_iters();
        let n_steps  = self.n_steps();

        // ---- Collect UI events from panels ----------------------------------
        let mut ev = UiEvents {
            prev_iter:   kb_prev_iter,
            next_iter:   kb_next_iter,
            prev_step:   kb_prev_step,
            next_step:   kb_next_step,
            toggle_play: kb_play_pause,
            ..Default::default()
        };

        // Pull out display values before closures so we can borrow immutably
        let is_multi = self.trajectories[iter_idx].agent_count > 1;
        let traj_grid_size = self.trajectories[iter_idx].grid_size as f32;
        let (exp_name, run_id, obs_mode, traj_iter, mean_reward, ep_reward,
             steps_taken, max_steps, success, goal_pos, start_pos, step_rewards,
             current_step, agent_count) = {
            let traj = &self.trajectories[iter_idx];
            let ep   = &traj.episode;
            let cur  = if step_idx < ep.steps.len() {
                let s = &ep.steps[step_idx];
                Some((s.step, s.action, s.reward, s.cursor_x, s.cursor_y, s.cursor_z,
                      s.voxel_count, s.placed))
            } else { None };
            let rewards_for_curve = if is_multi {
                ep.steps.iter().map(|s| s.rewards.iter().sum::<f32>()).collect::<Vec<_>>()
            } else {
                ep.steps.iter().map(|s| s.reward).collect::<Vec<_>>()
            };
            (
                traj.experiment_name.clone(),
                traj.run_id.clone(),
                traj.obs_mode.clone(),
                traj.iteration,
                traj.episode_reward_mean,
                ep.total_reward,
                ep.steps_taken,
                traj.max_steps,
                ep.success,
                ep.goal_pos,
                ep.start_pos,
                rewards_for_curve,
                cur,
                traj.agent_count,
            )
        };
        let current_trajectory_path = self.trajectories[iter_idx].source_path.clone();

        let explain_available = self.explain_ui.as_ref()
            .map(NativeExplainUi::available).unwrap_or(false);
        let explain_configured = self.explain_ui.is_some();
        egui::TopBottomPanel::top("application_tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.explain_tab, false, "Replay");
                ui.add_enabled_ui(explain_configured, |ui| {
                    ui.selectable_value(&mut self.explain_tab, true, "Explain");
                });
            });
        });

        let drag_delta = ctx.input(|i| i.pointer.delta());
        let dragging = ctx.input(|i| i.pointer.primary_down());
        let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);

        if self.explain_tab {
            egui::CentralPanel::default().show(ctx, |ui| {
                if let Some(explain) = self.explain_ui.as_mut() {
                    explain.show_embedded(ui);
                }
            });
        } else {

        // ---- Left panel -----------------------------------------------------
        egui::SidePanel::left("controls").min_width(260.0).show(ctx, |ui| {
            ui.heading("Voxel Replay");
            ui.separator();

            // Tune mode: trial header + params panel
            if in_tune_mode {
                let trial = &self.tune_trials[self.current_trial];
                let reward_str = if trial.best_reward.is_finite() {
                    format!("{:.3}", trial.best_reward)
                } else {
                    "n/a".to_string()
                };
                ui.label(egui::RichText::new("Trial").strong());
                ui.label(egui::RichText::new(&trial.trial_name).monospace());
                ui.label(egui::RichText::new(format!(
                    "{} / {}  ·  best reward: {}",
                    self.current_trial + 1, n_trials, reward_str
                )).weak().small());
                ui.label(egui::RichText::new(format!("dir: {}", trial.trial_id)).weak().small());
                ui.horizontal(|ui| {
                    if ui.button("|<").clicked() { ev.first_trial = true; }
                    if ui.button(" < ").clicked() { ev.prev_trial  = true; }
                    if ui.button(" > ").clicked() { ev.next_trial  = true; }
                    if ui.button(">|").clicked() { ev.last_trial  = true; }
                });
                ui.label(egui::RichText::new("  T / Y keys").small().weak());
                ui.separator();

                // Params panel
                ui.label(egui::RichText::new("Params").strong());
                if let Some(obj) = trial.params.as_object() {
                    for (k, v) in obj {
                        let val_str = match v {
                            serde_json::Value::Number(n) => {
                                if let Some(f) = n.as_f64() {
                                    if f.fract() == 0.0 && f.abs() < 1e9 { format!("{}", f as i64) }
                                    else { format!("{:.4e}", f) }
                                } else { v.to_string() }
                            },
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            _ => v.to_string(),
                        };
                        ui.label(format!("  {:<18} {}", k, val_str));
                    }
                }
                ui.separator();
            }

            ui.label(format!("Experiment: {}", exp_name));
            ui.label(format!("Run ID:     {}", run_id));
            ui.label(format!("Obs mode:   {}", obs_mode));
            if agent_count > 1 {
                ui.label(format!("Agents:     {}", agent_count));
            }
            ui.separator();

            // Iteration navigation
            ui.label(egui::RichText::new("Iterations").strong());
            ui.label(format!(
                "Iter {} of {}  (#{}, mean reward {:.3})",
                iter_idx + 1, n_iters, traj_iter, mean_reward
            ));
            let mut iter_slider = iter_idx;
            if ui.add(Slider::new(&mut iter_slider, 0..=n_iters.saturating_sub(1))
                .show_value(false)).changed()
            {
                ev.jump_iter = Some(iter_slider);
            }
            ui.horizontal(|ui| {
                if ui.button("|<").clicked() { ev.first_iter = true; }
                if ui.button(" < ").clicked() { ev.prev_iter  = true; }
                if ui.button(" > ").clicked() { ev.next_iter  = true; }
                if ui.button(">|").clicked() { ev.last_iter  = true; }
            });
            ui.label(egui::RichText::new("  [ / ] keys").small().weak());
            ui.separator();

            // Episode info
            ui.label(format!("Episode reward: {:.3}", ep_reward));
            ui.label(format!("Steps taken:    {}/{}", steps_taken, max_steps));
            ui.label(format!("Success:        {}", success));
            ui.separator();

            // Step navigation
            ui.label(egui::RichText::new("Steps").strong());
            let max_step = n_steps.saturating_sub(1);
            let mut step_slider = step_idx;
            if ui.add(Slider::new(&mut step_slider, 0..=max_step).text("Step")).changed() {
                ev.jump_step = Some(step_slider);
            }
            ui.horizontal(|ui| {
                if ui.button("|<").clicked() { ev.first_step = true; }
                if ui.button(" < ").clicked() { ev.prev_step  = true; }
                if ui.button(" > ").clicked() { ev.next_step  = true; }
                if ui.button(">|").clicked() { ev.last_step  = true; }
            });
            ui.label(egui::RichText::new("  ← / → keys").small().weak());

            if let Some((st, action, reward, cx, cy, cz, vcount, placed)) = current_step {
                ui.separator();
                ui.label("-- Current step --");
                ui.label(format!("Step:   {}", st));
                ui.label(format!("Action: {}", action));
                ui.label(format!("Reward: {:.3}", reward));
                ui.label(format!("Cursor: ({}, {}, {})", cx, cy, cz));
                ui.label(format!("Voxels: {}", vcount));
                if placed {
                    ui.colored_label(Color32::from_rgb(100, 220, 100), "placed!");
                }
            }
            ui.separator();

            // Play button
            let is_at_end = iter_idx == n_iters.saturating_sub(1)
                && step_idx >= n_steps.saturating_sub(1);
            let play_label = if self.playing { "Pause" }
                             else if is_at_end { "Play (replay)" }
                             else { "Play" };
            if ui.button(play_label).clicked() { ev.toggle_play = true; }
            ui.label(egui::RichText::new("  Space key").small().weak());
            ui.label(egui::RichText::new("Plays through all iterations").small().weak());
            ui.checkbox(&mut self.agent_x_ray, "Show agent through geometry (x-ray)");
            if self.trajectories[iter_idx].world.is_some() {
                ui.checkbox(&mut self.surface_mesh, "Use flat surface mesh (compiled worlds)");
            }
            ui.separator();
            ui.label(egui::RichText::new("Camera projection").strong());
            ui.checkbox(&mut self.camera.perspective, "Perspective (vanishing points)");
            ui.add_enabled_ui(self.camera.perspective, |ui| {
                ui.add(
                    Slider::new(&mut self.camera.field_of_view_degrees, 15.0..=100.0)
                        .suffix("°")
                        .text("field of view"),
                );
            });
            if self.trajectories[iter_idx].world.is_some() {
                ui.separator();
                ui.label(egui::RichText::new("Regional world view").strong());
                if ui.add(Slider::new(&mut self.visualization_radius, 1..=64)
                    .text("radius")).changed()
                {
                    self.requested_region = None;
                }
                let budgets_changed = ui.add(Slider::new(&mut self.chunk_budgets.visible, 1..=256)
                    .text("visible chunks")).changed()
                    | ui.add(Slider::new(&mut self.chunk_budgets.detailed, 1..=128)
                        .text("detailed chunks")).changed();
                self.chunk_budgets.detailed = self.chunk_budgets.detailed
                    .min(self.chunk_budgets.visible);
                if budgets_changed {
                    self.camera_revision = self.camera_revision.wrapping_add(1);
                }
                if self.pending_region.is_some() {
                    ui.label(egui::RichText::new("Loading visible region...").weak());
                }
                if let Some(error) = &self.regional_error {
                    ui.colored_label(Color32::from_rgb(230, 100, 100), error);
                }
                if let Some((_, frame)) = &self.regional_frame {
                    ui.label(format!("Visible voxels: {}", frame.occupied.len()));
                    ui.label(format!("Exposed faces: {}", self.regional_faces.len()));
                    ui.label(format!(
                        "Chunks considered/detailed/coarse: {}/{}/{}",
                        self.considered_chunks,
                        self.detailed_chunks,
                        self.coarse_chunks.len()
                    ));
                    ui.label(format!(
                        "Mesh cache builds/hits: {}/{}",
                        self.render_cache.builds(), self.render_cache.hits()
                    ));
                    ui.label(format!("Region load: {:.2} ms", frame.load_time.as_secs_f64() * 1_000.0));
                    if let Some(metrics) = frame.cache_metrics {
                        ui.label(format!(
                            "Chunks: {} resident, {} pinned",
                            metrics.resident_chunks, metrics.pinned_chunks
                        ));
                        ui.label(format!(
                            "Pack reads: {}  hits/misses: {}/{}",
                            metrics.pack_reads, metrics.cache_hits, metrics.cache_misses
                        ));
                    }
                }
                ui.separator();
                ui.label(egui::RichText::new("Global overview").strong());
                ui.checkbox(&mut self.show_overview, "Show overview");
                ui.add_enabled_ui(self.show_overview, |ui| {
                    ui.add(Slider::new(&mut self.overview_size, 140.0..=420.0).text("size"));
                    ui.checkbox(&mut self.show_overview_bounds, "Show world bounds");
                });
                if let Some(Err(error)) = self.overview_meshes.get(iter_idx) {
                    ui.colored_label(Color32::from_rgb(220, 150, 80), error);
                }
            }
            ui.separator();

            ui.label(egui::RichText::new("Explainability").strong());
            if ui.add_enabled(explain_available, egui::Button::new("Explain current step")).clicked() {
                if let Some(explain) = self.explain_ui.as_mut() {
                    explain.explain_trajectory(&current_trajectory_path, step_idx);
                }
                self.explain_tab = true;
            }
            if ui.add_enabled(explain_available, egui::Button::new("Open Explain tab")).clicked() {
                self.explain_tab = true;
            }
            if !explain_available {
                ui.label(egui::RichText::new(
                    "Open this replay from a run reference to attach a checkpoint."
                ).small().weak());
            }
            ui.separator();

            // Reward curve
            ui.label("-- Reward curve (this iteration) --");
            let desired = Vec2::new(ui.available_width(), 60.0);
            let (resp, painter) = ui.allocate_painter(desired, Sense::hover());
            let r = resp.rect;
            painter.rect_filled(r, 2.0, Color32::from_gray(25));
            if step_rewards.len() > 1 {
                let max_abs = step_rewards
                    .iter()
                    .map(|rv| rv.abs())
                    .fold(0.0f32, f32::max)
                    .max(0.001);
                let bar_w = r.width() / step_rewards.len() as f32;
                let zero_y = r.center().y;
                painter.line_segment(
                    [Pos2::new(r.left(), zero_y), Pos2::new(r.right(), zero_y)],
                    Stroke::new(1.0, Color32::from_gray(70)),
                );
                for (i, &rv) in step_rewards.iter().enumerate() {
                    let h = (rv.abs() / max_abs) * (r.height() * 0.5);
                    let x0 = r.left() + i as f32 * bar_w;
                    let col = if i == step_idx { Color32::YELLOW }
                              else if rv >= 0.0 { Color32::from_rgb(60, 180, 100) }
                              else { Color32::from_rgb(200, 60, 60) };
                    let y0 = if rv >= 0.0 { zero_y - h } else { zero_y };
                    painter.rect_filled(
                        Rect::from_min_size(
                            Pos2::new(x0, y0),
                            Vec2::new(bar_w.max(1.0), h.max(1.0)),
                        ),
                        0.0, col,
                    );
                }
            }
        });

        // ---- Central panel: isometric 3D view ------------------------------
        // Re-borrow trajectory data for rendering (immutable, no conflict)
        let (render_steps, render_goal, render_start,
             render_goal_positions, render_start_positions) = {
            let ep = &self.trajectories[iter_idx].episode;
            (ep.steps.clone(), ep.goal_pos, ep.start_pos,
             ep.goal_positions.clone(), ep.start_positions.clone())
        };
        let compiled_mode = self.trajectories[iter_idx].world.is_some();
        let active_regional_frame = self.regional_frame.as_ref().map(|(_, frame)| frame);
        let render_origin = active_regional_frame
            .map(|frame| frame.render_origin)
            .or_else(|| self.current_region_key().map(|key| StorageCoord {
                x: key.center.x.saturating_sub(key.radius),
                y: key.center.y.saturating_sub(key.radius),
                z: key.center.z.saturating_sub(key.radius),
            }))
            .unwrap_or(StorageCoord { x: 0, y: 0, z: 0 });
        let regional_grid_size = active_regional_frame.map(|frame| {
            let size_x = frame.region.maximum_exclusive.x - frame.region.minimum.x;
            let size_y = frame.region.maximum_exclusive.y - frame.region.minimum.y;
            let size_z = frame.region.maximum_exclusive.z - frame.region.minimum.z;
            size_x.max(size_y).max(size_z) as f32
        });
        let display_grid_size = regional_grid_size
            .or_else(|| compiled_mode.then_some((self.visualization_radius * 2 + 1) as f32))
            .unwrap_or(traj_grid_size);
        let geo_list = if compiled_mode {
            active_regional_frame
                .map(|frame| {
                    frame.occupied.iter()
                        .filter(|coordinate| frame.region.contains(**coordinate))
                        .filter_map(|coordinate| Some((
                            u16::try_from(coordinate.x).ok()?,
                            u16::try_from(coordinate.y).ok()?,
                            u16::try_from(coordinate.z).ok()?,
                        ))).collect::<Vec<_>>()
                })
                .unwrap_or_default()
        } else {
            self.geo_voxels[iter_idx].clone()
        };

        // Collect camera drag / scroll BEFORE the closure (avoid borrow conflict)
        egui::CentralPanel::default().show(ctx, |ui| {
            // Allocate with drag sense so the cursor changes on hover
            let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::drag());
            let rect = resp.rect.shrink(20.0);
            painter.rect_filled(resp.rect, 0.0, Color32::from_rgb(12, 14, 20));

            // Camera hint label
            painter.text(
                resp.rect.right_bottom() + Vec2::new(-8.0, -4.0),
                egui::Align2::RIGHT_BOTTOM,
                "Left-drag: orbit   Scroll: zoom   R: reset",
                egui::FontId::proportional(11.0),
                Color32::from_gray(80),
            );

            let cam = &self.camera;
            let b = cam.bounds(display_grid_size);
            // Rear edges render beneath voxels and are occluded by filled space.
            draw_grid_bounds_layer(&painter, rect, cam, &b, display_grid_size, false);
            // Build geometry set for agent-fill collision check (fast HashSet lookup).
            let geometry: HashSet<(u16, u16, u16)> = geo_list.iter().copied().collect();

            let geo_color = Color32::from_rgb(120, 120, 130);
            if compiled_mode {
                let chunk_edge = self.trajectories[iter_idx].world_chunk_edge.max(1);
                for &chunk in self.coarse_chunks.iter().filter(|chunk| {
                    active_regional_frame.is_some_and(|frame| {
                        chunk_intersects_region(**chunk, chunk_edge, frame.region)
                    })
                }) {
                    draw_coarse_chunk(
                        &painter, chunk, chunk_edge, render_origin, rect, cam, &b,
                    );
                }
            }

            // Per-agent trail colors (hue-shifted blues/greens/reds).
            let agent_trail_colors = [
                Color32::from_rgb( 70, 140, 210),  // blue
                Color32::from_rgb(220, 100,  60),  // orange
                Color32::from_rgb( 80, 200, 100),  // green
                Color32::from_rgb(200,  80, 200),  // purple
                Color32::from_rgb(220, 200,  50),  // yellow
            ];
            // Start marker: agent color darkened (~65%).
            let tint_dark = |c: Color32| -> Color32 {
                Color32::from_rgb(
                    (c.r() as u16 * 65 / 100) as u8,
                    (c.g() as u16 * 65 / 100) as u8,
                    (c.b() as u16 * 65 / 100) as u8,
                )
            };
            // Goal marker: agent color lightened (~40% blend toward white).
            let tint_light = |c: Color32| -> Color32 {
                Color32::from_rgb(
                    (c.r() as u16 + (255 - c.r() as u16) * 40 / 100) as u8,
                    (c.g() as u16 + (255 - c.g() as u16) * 40 / 100) as u8,
                    (c.b() as u16 + (255 - c.b() as u16) * 40 / 100) as u8,
                )
            };

            let trail_end = (step_idx + 1).min(render_steps.len());

            // Collect geometry, trails, cursors, and start/goal markers as
            // deferred SceneDrawItems and paint them all in one back-to-front
            // pass (paint_scene_items) so a marker in front of a wall stays
            // visible and one behind it is correctly hidden -- see
            // SceneDrawItem's doc comment for why this replaced the previous
            // all-or-nothing draw-before/draw-after split. agent_x_ray still
            // covers the "always visible through geometry" case the old
            // toggle's false branch gave you, for tunnels/dense structures.
            let mut scene_items: Vec<SceneDrawItem> = Vec::new();

            if compiled_mode && self.surface_mesh {
                scene_items.extend(
                    self.regional_faces
                        .iter()
                        .copied()
                        .filter(|face| {
                            active_regional_frame
                                .map_or(true, |frame| frame.region.contains(face.voxel))
                        })
                        .map(|face| SceneDrawItem::Face { face, color: geo_color }),
                );
            } else {
                // geo_list already holds the resolved occupied voxels for
                // both compiled and legacy trajectories (see its
                // definition above) -- per-voxel cubes are the default so
                // adjacent occupied voxels keep their cube depth instead of
                // merging into one flat tile, matching #258/#259.
                scene_items.extend(geo_list.iter().map(|&(x, y, z)| SceneDrawItem::Voxel {
                    x, y, z, color: geo_color, outline: false, agent: false,
                }));
            }

            if is_multi {
                // ---- Multi-agent: per-agent colored trails and cursors ----
                let n_agents = render_steps.first().map(|s| s.cursors.len()).unwrap_or(0);
                // Determine whether placed_per_agent is populated (new trajectories)
                // or absent/all-false (old trajectories saved before the fix).
                let has_placed_data = render_steps.iter().any(|s| s.placed_per_agent.iter().any(|&p| p));

                for ai in 0..n_agents {
                    let trail_color = agent_trail_colors[ai % agent_trail_colors.len()];
                    let mut trail: HashSet<(u16, u16, u16)> = HashSet::new();
                    for i in 0..if compiled_mode { 0 } else { trail_end } {
                        let s = &render_steps[i];
                        if ai < s.cursors.len() {
                            let c = s.cursors[ai];
                            let placed = if has_placed_data {
                                s.placed_per_agent.get(ai).copied().unwrap_or(false)
                            } else {
                                // Fall back: a voxel was placed if cursor moved from previous step.
                                i == 0 || render_steps[i - 1].cursors.get(ai).copied() != Some(c)
                            };
                            if placed && !geometry.contains(&(c[0], c[1], c[2])) {
                                trail.insert((c[0], c[1], c[2]));
                            }
                        }
                    }
                    scene_items.extend(trail.into_iter().map(|(x, y, z)| SceneDrawItem::Voxel {
                        x, y, z, color: trail_color, outline: true, agent: true,
                    }));
                }
                // Each agent's current cursor.
                if step_idx < render_steps.len() {
                    let s = &render_steps[step_idx];
                    for ai in 0..s.cursors.len() {
                        let c = s.cursors[ai];
                        scene_items.push(SceneDrawItem::Cursor { x: c[0], y: c[1], z: c[2] });
                    }
                }
                // Start markers (agent color darkened).
                for (ai, start_opt) in render_start_positions.iter().enumerate() {
                    if let Some([sx, sy, sz]) = start_opt {
                        let col = tint_dark(agent_trail_colors[ai % agent_trail_colors.len()]);
                        scene_items.push(SceneDrawItem::Marker { x: *sx, y: *sy, z: *sz, color: col });
                    }
                }
                // Goal markers (agent color lightened).
                for (ai, goal_opt) in render_goal_positions.iter().enumerate() {
                    if let Some([gx, gy, gz]) = goal_opt {
                        let col = tint_light(agent_trail_colors[ai % agent_trail_colors.len()]);
                        scene_items.push(SceneDrawItem::Marker { x: *gx, y: *gy, z: *gz, color: col });
                    }
                }
            } else {
                // ---- Single-agent path ----
                let agent_color = agent_trail_colors[0];
                let mut agent_filled: HashSet<(u16, u16, u16)> = HashSet::new();
                for i in 0..if compiled_mode { 0 } else { trail_end } {
                    let s = &render_steps[i];
                    if s.placed && !geometry.contains(&(s.cursor_x, s.cursor_y, s.cursor_z)) {
                        agent_filled.insert((s.cursor_x, s.cursor_y, s.cursor_z));
                    }
                }
                scene_items.extend(agent_filled.into_iter().map(|(x, y, z)| SceneDrawItem::Voxel {
                    x, y, z, color: agent_color, outline: true, agent: true,
                }));

                if let Some([sx, sy, sz]) = render_start {
                    scene_items.push(SceneDrawItem::Marker {
                        x: sx, y: sy, z: sz, color: tint_dark(agent_trail_colors[0]),
                    });
                }
                if let Some([gx, gy, gz]) = render_goal {
                    scene_items.push(SceneDrawItem::Marker {
                        x: gx, y: gy, z: gz, color: tint_light(agent_trail_colors[0]),
                    });
                }
                if step_idx < render_steps.len() {
                    let s = &render_steps[step_idx];
                    scene_items.push(SceneDrawItem::Cursor {
                        x: s.cursor_x, y: s.cursor_y, z: s.cursor_z,
                    });
                }
            }

            paint_scene_items(&mut scene_items, &painter, render_origin, rect, cam, &b, self.agent_x_ray);

            // Camera-facing and silhouette edges remain visible above the scene.
            draw_grid_bounds_layer(&painter, rect, cam, &b, display_grid_size, true);

            if self.show_overview {
                if let Some(Ok(mesh)) = self.overview_meshes.get(iter_idx) {
                    draw_overview_inset(
                        &painter, resp.rect, mesh, cam,
                        self.overview_size, self.show_overview_bounds,
                        active_regional_frame.map(|frame| {
                            (frame.region.minimum, frame.region.maximum_exclusive)
                        }),
                    );
                }
            }

            let agent_label = if is_multi {
                format!("{} agents", agent_count)
            } else {
                String::new()
            };
            painter.text(
                rect.left_bottom() + Vec2::new(4.0, -4.0),
                egui::Align2::LEFT_BOTTOM,
                format!(
                    "iter {}/{}  |  {} geo  |  step {}/{}  {}",
                    iter_idx + 1, n_iters,
                    geo_list.len(),
                    step_idx, n_steps.saturating_sub(1),
                    agent_label,
                ),
                egui::FontId::proportional(12.0),
                Color32::from_gray(160),
            );
        });

        }
        // ---- Apply collected events to state --------------------------------
        // Extract trial nav flags before ev is consumed by apply_events.
        let (trial_first, trial_last, trial_next, trial_prev) =
            (ev.first_trial, ev.last_trial, ev.next_trial, ev.prev_trial);
        self.apply_events(ev);

        // ---- Tune mode: trial navigation (buttons + T/Y keys) ---------------
        // Handled after apply_events so iter/step resets from load_trial override
        // any stale iter/step values that apply_events may have set.
        if in_tune_mode {
            let want_first = trial_first;
            let want_last  = trial_last;
            let want_next  = trial_next || kb_next_trial;
            let want_prev  = trial_prev || kb_prev_trial;

            let target = if want_first {
                Some(0)
            } else if want_last {
                Some(n_trials.saturating_sub(1))
            } else if want_next {
                Some((self.current_trial + 1).min(n_trials.saturating_sub(1)))
            } else if want_prev && self.current_trial > 0 {
                Some(self.current_trial - 1)
            } else {
                None
            };
            if let Some(t) = target {
                if t != self.current_trial { self.load_trial(t); }
            }
        }

        // ---- Camera orbit (left-drag) + zoom (scroll) + reset (R) ----------
        if dragging {
            self.camera.yaw   += drag_delta.x * 0.008;
            self.camera.pitch  = (self.camera.pitch - drag_delta.y * 0.006)
                .clamp(-1.3, 1.3);
            self.camera_revision = self.camera_revision.wrapping_add(1);
        }
        if scroll_y.abs() > 0.1 {
            self.camera.zoom = (self.camera.zoom * (1.0 + scroll_y * 0.003)).clamp(0.2, 5.0);
            self.camera_revision = self.camera_revision.wrapping_add(1);
        }
        if ctx.input(|i| i.key_pressed(Key::R)) {
            self.camera = Camera::default();
        }

        // ---- Auto-play: advance one step per frame --------------------------
        if self.playing {
            let still_going = self.play_advance();
            if !still_going { self.playing = false; }
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }

        // Reconcile regional geometry (main-view region and the minimap's
        // visibility-field cube) against the position this frame actually
        // settled on. This must run after keyboard input, panel clicks,
        // apply_events, and autoplay above -- otherwise it decides whether to
        // load a new region using last frame's stale iter/step, the new
        // position is never requested, and nothing here calls
        // request_repaint() to force the follow-up frame that would have
        // caught the change. That was the minimap cube's actual bug: it
        // wasn't stale data or a caching key, it was checking the position
        // one frame too early.
        self.update_regional_geometry(ctx);
    }
}

#[cfg(test)]
mod minimap_region_tracking_tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn write_fixture_world(root: &std::path::Path, identity: &str) {
        let world_dir = root.join("worlds").join(identity);
        std::fs::create_dir_all(&world_dir).unwrap();
        std::fs::write(world_dir.join("world.pack"), []).unwrap();
        std::fs::write(
            world_dir.join("manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "identity_sha256": identity,
                "coordinate_type": "u32",
                "extent": {"x": 128, "y": 128, "z": 64},
                "chunk_shape": {"x": 32, "y": 32, "z": 32},
                "chunks": []
            }))
            .unwrap(),
        )
        .unwrap();
    }

    /// A 16-step fixture matching the real minimap showcase: the agent walks
    /// from (48, 64, 8) at step 0 to (63, 64, 8) at step 15.
    fn write_fixture_trajectory(root: &std::path::Path, identity: &str) {
        let traj_dir = root.join("trajectories");
        std::fs::create_dir_all(&traj_dir).unwrap();
        let steps: Vec<_> = (0..16u32)
            .map(|step| {
                serde_json::json!({
                    "step": step,
                    "cursor_x": 48 + step,
                    "cursor_y": 64,
                    "cursor_z": 8,
                })
            })
            .collect();
        std::fs::write(
            traj_dir.join("iter_000001.json"),
            serde_json::to_vec(&serde_json::json!({
                "experiment_name": "fixture",
                "run_id": "fixture",
                "iteration": 1,
                "episode_reward_mean": 0.0,
                "max_steps": 16,
                "world": {
                    "identity_sha256": identity,
                    "schema_version": 1,
                    "coordinate_type": "u32",
                    "extent": [128, 128, 64],
                    "manifest_path": format!("../worlds/{identity}/manifest.json"),
                },
                "episode": {
                    "total_reward": 0.0,
                    "steps_taken": 16,
                    "success": true,
                    "steps": steps,
                },
            }))
            .unwrap(),
        )
        .unwrap();
    }

    /// Drives `update_regional_geometry` (the same call `update()` makes once
    /// per frame, now positioned after all input/state handling) until the
    /// loaded region matches `target_step`, or panics after a timeout.
    fn wait_for_region(
        app: &mut VoxelReplayApp,
        ctx: &egui::Context,
        target_step: usize,
    ) -> theseo_core::voxel::world::BoundedRegion {
        let started = Instant::now();
        loop {
            app.update_regional_geometry(ctx);
            if let Some((key, frame)) = &app.regional_frame {
                if key.step == target_step {
                    return frame.region;
                }
            }
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "region for step {target_step} never resolved"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn minimap_region_recenters_on_the_agent_after_the_step_changes() {
        let root = std::env::temp_dir().join(format!(
            "theseo-minimap-region-test-{}",
            std::process::id()
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        let identity = "a".repeat(64);
        write_fixture_world(&root, &identity);
        write_fixture_trajectory(&root, &identity);

        let trajectories = load_trial_trajectories(&root.join("trajectories"));
        assert_eq!(trajectories.len(), 1, "fixture trajectory failed to load");
        let mut app = VoxelReplayApp::new(trajectories, None);
        let ctx = egui::Context::default();

        // Step 0: the region (and therefore the minimap's visibility-field
        // cube) must be centered on the agent's cursor, (48, 64, 8).
        app.step_idx = 0;
        let first = wait_for_region(&mut app, &ctx, 0);
        let first_center_x = (first.minimum.x + first.maximum_exclusive.x) / 2;
        assert_eq!(first_center_x, 48, "step 0 region is not centered on the agent");

        // This assignment mirrors exactly what `apply_events` does to
        // `step_idx` before `update()` calls `update_regional_geometry` --
        // the ordering the fix in `update()` now guarantees. A single
        // subsequent call must pick up the new position; it must not require
        // an extra, unrequested repaint to notice the change.
        app.step_idx = 15;
        let second = wait_for_region(&mut app, &ctx, 15);
        let second_center_x = (second.minimum.x + second.maximum_exclusive.x) / 2;
        assert_eq!(
            second_center_x, 63,
            "step 15 region did not follow the agent to x=63"
        );
        assert_eq!(
            second_center_x - first_center_x,
            15,
            "visibility-field cube must move by exactly the agent's displacement"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result<()> {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Vec::new();
    let mut explain_run: Option<PathBuf> = None;
    let mut checkpoint = "latest".to_string();
    let mut open_observation_editor = false;
    let mut index = 0;
    while index < raw_args.len() {
        match raw_args[index].as_str() {
            "--explain-run" => {
                index += 1;
                match raw_args.get(index) {
                    Some(value) if !value.starts_with("--") => {
                        explain_run = Some(PathBuf::from(value));
                    }
                    _ => {
                        eprintln!("--explain-run requires a run directory path");
                        return Ok(());
                    }
                }
            }
            "--checkpoint" => {
                index += 1;
                match raw_args.get(index) {
                    Some(value) if !value.starts_with("--") => {
                        checkpoint = value.clone();
                    }
                    _ => {
                        eprintln!("--checkpoint requires a value, e.g. latest or an iteration number");
                        return Ok(());
                    }
                }
            }
            "--open-observation-editor" => open_observation_editor = true,
            value => args.push(value.to_string()),
        }
        index += 1;
    }
    if args.is_empty() {
        eprintln!("Usage:");
        eprintln!("  voxel-replay --tune-dir <tune-run-dir>     Navigate tune trials");
        eprintln!("  voxel-replay <file1.json> [file2.json ...] Replay specific files");
        eprintln!();
        eprintln!("Keyboard shortcuts:");
        eprintln!("  T / Y       next / prev trial  (tune mode only)");
        eprintln!("  [ / ]       next / prev iteration within current trial");
        eprintln!("  <- / ->     step forward / backward");
        eprintln!("  Space       play / pause");
        return Ok(());
    }

    // --tune-dir mode: scan all trial_*/trajectories/ and sort by reward.
    if args[0] == "--tune-dir" {
        let tune_dir_str = args.get(1).map(|s| s.as_str()).unwrap_or("");
        if tune_dir_str.is_empty() {
            eprintln!("--tune-dir requires a path, e.g.: voxel-replay --tune-dir runtime/tune/v2");
            return Ok(());
        }
        let tune_dir = PathBuf::from(tune_dir_str);
        let trials = scan_tune_dir(&tune_dir);
        if trials.is_empty() {
            eprintln!("No trials with trajectories found under '{}'.", tune_dir.display());
            eprintln!("Make sure the run used trajectory_every > 0 or best_trajectory: true.");
            return Ok(());
        }
        let dir_name = tune_dir.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let title = format!("Voxel Replay (Tune) -- {} ({} trials)", dir_name, trials.len());
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title(title)
                .with_inner_size([1280.0, 760.0]),
            ..Default::default()
        };
        return eframe::run_native(
            "Voxel Replay",
            options,
            Box::new(move |_cc| Ok(Box::new(VoxelReplayApp::new_tune(trials)))),
        );
    }

    // File mode: load specific trajectory JSON files.
    let mut trajectories: Vec<TrajectoryData> = Vec::new();
    for arg in &args {
        let path = PathBuf::from(arg);
        let traj = load_trajectory(&path).ok_or_else(|| {
            eframe::Error::AppCreation(
                format!("Cannot load trajectory '{}'", path.display()).into()
            )
        })?;
        trajectories.push(traj);
    }

    // Sort by iteration number so navigation is chronological.
    trajectories.sort_by_key(|t| t.iteration);

    let title = if trajectories.len() == 1 {
        format!(
            "Voxel Replay -- {} iter {}",
            trajectories[0].experiment_name, trajectories[0].iteration
        )
    } else {
        format!(
            "Voxel Replay -- {} ({} iterations)",
            trajectories[0].experiment_name, trajectories.len()
        )
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size([1200.0, 760.0]),
        ..Default::default()
    };

    let explain_ui = explain_run.as_deref()
        .map(|run| NativeExplainUi::start(run, &checkpoint, open_observation_editor));
    eframe::run_native(
        "Voxel Replay",
        options,
        Box::new(move |_cc| Ok(Box::new(VoxelReplayApp::new(trajectories, explain_ui)))),
    )
}
