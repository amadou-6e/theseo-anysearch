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
use std::collections::HashSet;
use std::path::PathBuf;

use eframe::egui::{self, Color32, Key, Pos2, Rect, Sense, Shape, Slider, Stroke, Vec2};
use serde::Deserialize;

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

// ---------------------------------------------------------------------------
// Camera: holds yaw/pitch/zoom and handles projection + screen mapping
// ---------------------------------------------------------------------------

struct Camera {
    yaw:   f32,   // radians — rotates x/z plane
    pitch: f32,   // radians — tilts up/down
    zoom:  f32,   // scale multiplier (1.0 = default)
}

impl Camera {
    fn default() -> Self {
        Self { yaw: 45.0_f32.to_radians(), pitch: 30.0_f32.to_radians(), zoom: 1.0 }
    }

    /// Project a 3-D grid coord to 2-D screen space (pre-bounds).
    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let xr  = x * self.yaw.cos() - z * self.yaw.sin();
        let zr  = x * self.yaw.sin() + z * self.yaw.cos();
        let yr  = y * self.pitch.cos() - zr * self.pitch.sin();
        let zr2 = y * self.pitch.sin() + zr * self.pitch.cos();
        (xr, -(yr - zr2 * 0.05))
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
        Bounds { min_x: min_x - pw, max_x: max_x + pw, min_y: min_y - ph, max_y: max_y + ph }
    }

    /// Map a 3-D coord to a pixel position inside `rect`, applying zoom.
    #[inline]
    fn to_screen(&self, x: f32, y: f32, z: f32, rect: Rect, b: &Bounds) -> Pos2 {
        let (px, py) = self.project(x, y, z);
        let cx = rect.center().x;
        let cy = rect.center().y;
        let w  = rect.width()  * self.zoom;
        let h  = rect.height() * self.zoom;
        Pos2::new(
            cx + (px - (b.min_x + b.max_x) * 0.5) / (b.max_x - b.min_x).max(1.0) * w,
            cy + (py - (b.min_y + b.max_y) * 0.5) / (b.max_y - b.min_y).max(1.0) * h,
        )
    }
}

struct Bounds { min_x: f32, max_x: f32, min_y: f32, max_y: f32 }

/// Back-to-front depth key for painter's algorithm.
/// Voxels with lower value are further from camera and should be drawn first.
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

fn draw_cursor(painter: &egui::Painter, cx: u16, cy: u16, cz: u16,
               rect: Rect, cam: &Camera, b: &Bounds) {
    let (x, y, z) = (cx as f32, cy as f32, cz as f32);
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

fn draw_marker(
    painter: &egui::Painter,
    cx: u16, cy: u16, cz: u16,
    rect: Rect, cam: &Camera, b: &Bounds,
    color: Color32,
) {
    draw_voxel(painter, cx, cy, cz, rect, cam, b, color, true);
    let (x, y, z) = (cx as f32, cy as f32, cz as f32);
    let h = 0.4_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);
    let stroke = Stroke::new(2.0, Color32::WHITE);
    painter.line_segment([corner(-h, h, -h), corner(h, h, h)], stroke);
    painter.line_segment([corner(h, h, -h), corner(-h, h, h)], stroke);
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
    /// Tune mode: all trials sorted best-first. Empty in file mode.
    tune_trials: Vec<TrialEntry>,
    /// Which trial is currently loaded (index into tune_trials).
    current_trial: usize,
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
    fn new(trajectories: Vec<TrajectoryData>) -> Self {
        let geo_voxels = trajectories.iter().map(|t| {
            t.episode.init_filled.iter().map(|c| (c[0], c[1], c[2])).collect()
        }).collect();
        Self {
            camera: Camera::default(),
            trajectories,
            geo_voxels,
            iter_idx: 0,
            step_idx: 0,
            playing: false,
            tune_trials: Vec::new(),
            current_trial: 0,
        }
    }

    fn new_tune(trials: Vec<TrialEntry>) -> Self {
        let mut app = Self {
            camera: Camera::default(),
            trajectories: Vec::new(),
            geo_voxels: Vec::new(),
            iter_idx: 0,
            step_idx: 0,
            playing: false,
            tune_trials: trials,
            current_trial: 0,
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
        self.trajectories = trajs;
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
        let geo_list = &self.geo_voxels[iter_idx];

        // Collect camera drag / scroll BEFORE the closure (avoid borrow conflict)
        let drag_delta = ctx.input(|i| i.pointer.delta());
        let dragging   = ctx.input(|i| i.pointer.primary_down());
        let scroll_y   = ctx.input(|i| i.smooth_scroll_delta.y);

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
            let b = cam.bounds(traj_grid_size);

            // Build geometry set for agent-fill collision check (fast HashSet lookup).
            let geometry: HashSet<(u16, u16, u16)> = geo_list.iter().copied().collect();

            let geo_color = Color32::from_rgb(120, 120, 130);

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

            // Sort geometry back-to-front.
            let mut geo_sorted: Vec<(u16, u16, u16)> = geo_list.iter().copied().collect();
            geo_sorted.sort_by(|a, b| {
                depth_key(a.0, a.1, a.2, cam)
                    .partial_cmp(&depth_key(b.0, b.1, b.2, cam))
                    .unwrap()
            });
            for &(x, y, z) in &geo_sorted { draw_voxel(&painter, x, y, z, rect, cam, &b, geo_color, false); }

            if is_multi {
                // ---- Multi-agent: per-agent colored trails and cursors ----
                let n_agents = render_steps.first().map(|s| s.cursors.len()).unwrap_or(0);
                // Determine whether placed_per_agent is populated (new trajectories)
                // or absent/all-false (old trajectories saved before the fix).
                let has_placed_data = render_steps.iter().any(|s| s.placed_per_agent.iter().any(|&p| p));

                for ai in 0..n_agents {
                    let trail_color = agent_trail_colors[ai % agent_trail_colors.len()];
                    let mut trail: HashSet<(u16, u16, u16)> = HashSet::new();
                    for i in 0..trail_end {
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
                    let mut trail_list: Vec<(u16, u16, u16)> = trail.into_iter().collect();
                    trail_list.sort_by(|a, b| {
                        depth_key(a.0, a.1, a.2, cam)
                            .partial_cmp(&depth_key(b.0, b.1, b.2, cam))
                            .unwrap()
                    });
                    for &(x, y, z) in &trail_list {
                        draw_voxel(&painter, x, y, z, rect, cam, &b, trail_color, true);
                    }
                }
                // Draw each agent's current cursor.
                if step_idx < render_steps.len() {
                    let s = &render_steps[step_idx];
                    for ai in 0..s.cursors.len() {
                        let c = s.cursors[ai];
                        draw_cursor(&painter, c[0], c[1], c[2], rect, cam, &b);
                    }
                }
                // Draw start markers (agent color darkened).
                for (ai, start_opt) in render_start_positions.iter().enumerate() {
                    if let Some([sx, sy, sz]) = start_opt {
                        let col = tint_dark(agent_trail_colors[ai % agent_trail_colors.len()]);
                        draw_marker(&painter, *sx, *sy, *sz, rect, cam, &b, col);
                    }
                }
                // Draw goal markers (agent color lightened).
                for (ai, goal_opt) in render_goal_positions.iter().enumerate() {
                    if let Some([gx, gy, gz]) = goal_opt {
                        let col = tint_light(agent_trail_colors[ai % agent_trail_colors.len()]);
                        draw_marker(&painter, *gx, *gy, *gz, rect, cam, &b, col);
                    }
                }
            } else {
                // ---- Single-agent path ----
                let agent_color = agent_trail_colors[0];
                let mut agent_filled: HashSet<(u16, u16, u16)> = HashSet::new();
                for i in 0..trail_end {
                    let s = &render_steps[i];
                    if s.placed && !geometry.contains(&(s.cursor_x, s.cursor_y, s.cursor_z)) {
                        agent_filled.insert((s.cursor_x, s.cursor_y, s.cursor_z));
                    }
                }
                let mut agent_list: Vec<(u16, u16, u16)> = agent_filled.into_iter().collect();
                agent_list.sort_by(|a, b| {
                    depth_key(a.0, a.1, a.2, cam)
                        .partial_cmp(&depth_key(b.0, b.1, b.2, cam))
                        .unwrap()
                });
                for &(x, y, z) in &agent_list {
                    draw_voxel(&painter, x, y, z, rect, cam, &b, agent_color, true);
                }

                if let Some([sx, sy, sz]) = render_start {
                    draw_marker(&painter, sx, sy, sz, rect, cam, &b, tint_dark(agent_trail_colors[0]));
                }
                if let Some([gx, gy, gz]) = render_goal {
                    draw_marker(&painter, gx, gy, gz, rect, cam, &b, tint_light(agent_trail_colors[0]));
                }
                if step_idx < render_steps.len() {
                    let s = &render_steps[step_idx];
                    draw_cursor(&painter, s.cursor_x, s.cursor_y, s.cursor_z, rect, cam, &b);
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
                    geo_sorted.len(),
                    step_idx, n_steps.saturating_sub(1),
                    agent_label,
                ),
                egui::FontId::proportional(12.0),
                Color32::from_gray(160),
            );
        });

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
        }
        if scroll_y.abs() > 0.1 {
            self.camera.zoom = (self.camera.zoom * (1.0 + scroll_y * 0.003)).clamp(0.2, 5.0);
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
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
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

    eframe::run_native(
        "Voxel Replay",
        options,
        Box::new(move |_cc| Ok(Box::new(VoxelReplayApp::new(trajectories)))),
    )
}
