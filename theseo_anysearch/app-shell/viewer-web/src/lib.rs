//! Voxel replay viewport, compiled to WebAssembly and mounted as a `<canvas>`
//! by the React shell. The camera/projection/painter code below is a trimmed,
//! faithful port of `voxel_replay.rs`'s `Camera`, `draw_voxel`, `draw_cursor`
//! and `depth_key` — same math, same draw order — with file I/O and the
//! Tune-trial/Explain-tab machinery left out (those stay native for now).

use std::cell::RefCell;
use std::rc::Rc;

use eframe::egui::{self, Color32, Pos2, Rect, Shape, Stroke};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

// ---------------------------------------------------------------------------
// Data model — mirrors the JSON trajectory format (see voxel_replay.rs).
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Default)]
pub struct StepData {
    #[serde(default)]
    pub cursor_x: u16,
    #[serde(default)]
    pub cursor_y: u16,
    #[serde(default)]
    pub cursor_z: u16,
    #[serde(default)]
    pub reward: f32,
    #[serde(default)]
    pub action: i32,
    #[serde(default)]
    pub voxel_count: u32,
    #[serde(default)]
    pub placed: bool,
}

#[derive(Deserialize, Clone, Default)]
pub struct EpisodeData {
    #[serde(default)]
    pub init_filled: Vec<[u16; 3]>,
    #[serde(default)]
    pub steps: Vec<StepData>,
    #[serde(default)]
    pub total_reward: f32,
    #[serde(default)]
    pub steps_taken: u32,
    #[serde(default)]
    pub success: bool,
}

#[derive(Deserialize, Clone, Default)]
pub struct TrajectoryData {
    #[serde(default)]
    pub iteration: u32,
    #[serde(default = "default_grid_size")]
    pub grid_size: u16,
    pub episode: EpisodeData,
}

fn default_grid_size() -> u16 {
    32
}

// ---------------------------------------------------------------------------
// Camera — identical projection math to the native viewer.
// ---------------------------------------------------------------------------

struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

impl Camera {
    fn default() -> Self {
        Self { yaw: 45.0_f32.to_radians(), pitch: 30.0_f32.to_radians(), zoom: 1.0 }
    }

    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let xr = x * self.yaw.cos() - z * self.yaw.sin();
        let zr = x * self.yaw.sin() + z * self.yaw.cos();
        let yr = y * self.pitch.cos() - zr * self.pitch.sin();
        let zr2 = y * self.pitch.sin() + zr * self.pitch.cos();
        (xr, -(yr - zr2 * 0.05))
    }

    fn bounds(&self, grid_size: f32) -> Bounds {
        let lo = 0.5f32;
        let hi = grid_size + 0.5;
        let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        for &xf in &[lo, hi] {
            for &yf in &[lo, hi] {
                for &zf in &[lo, hi] {
                    let (px, py) = self.project(xf, yf, zf);
                    min_x = min_x.min(px);
                    max_x = max_x.max(px);
                    min_y = min_y.min(py);
                    max_y = max_y.max(py);
                }
            }
        }
        let pw = (max_x - min_x) * 0.05;
        let ph = (max_y - min_y) * 0.05;
        Bounds { min_x: min_x - pw, max_x: max_x + pw, min_y: min_y - ph, max_y: max_y + ph }
    }

    #[inline]
    fn to_screen(&self, x: f32, y: f32, z: f32, rect: Rect, b: &Bounds) -> Pos2 {
        let (px, py) = self.project(x, y, z);
        let cx = rect.center().x;
        let cy = rect.center().y;
        let w = rect.width() * self.zoom;
        let h = rect.height() * self.zoom;
        Pos2::new(
            cx + (px - (b.min_x + b.max_x) * 0.5) / (b.max_x - b.min_x).max(1.0) * w,
            cy + (py - (b.min_y + b.max_y) * 0.5) / (b.max_y - b.min_y).max(1.0) * h,
        )
    }
}

struct Bounds {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

fn depth_key(x: u16, y: u16, z: u16, cam: &Camera) -> f32 {
    let (sy, cy) = (cam.yaw.sin(), cam.yaw.cos());
    let (sp, cp) = (cam.pitch.sin(), cam.pitch.cos());
    x as f32 * sy * cp + y as f32 * sp + z as f32 * cy * cp
}

fn draw_voxel(painter: &egui::Painter, cx: u16, cy: u16, cz: u16, rect: Rect, cam: &Camera, b: &Bounds, base: Color32) {
    let (x, y, z) = (cx as f32, cy as f32, cz as f32);
    let h = 0.5_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);

    let hx = if cam.yaw.sin() > 0.0 { h } else { -h };
    let hy = if cam.pitch.sin() > 0.0 { h } else { -h };
    let hz = if cam.yaw.cos() > 0.0 { h } else { -h };

    let top_face = vec![corner(-h, hy, -h), corner(h, hy, -h), corner(h, hy, h), corner(-h, hy, h)];
    let face_x = vec![corner(hx, -h, -h), corner(hx, -h, h), corner(hx, h, h), corner(hx, h, -h)];
    let face_z = vec![corner(-h, -h, hz), corner(h, -h, hz), corner(h, h, hz), corner(-h, h, hz)];

    let shade = |r: u8, g: u8, bl: u8, amount: i32| -> Color32 {
        Color32::from_rgb(
            (r as i32 + amount).clamp(0, 255) as u8,
            (g as i32 + amount).clamp(0, 255) as u8,
            (bl as i32 + amount).clamp(0, 255) as u8,
        )
    };
    let (r, g, bl) = (base.r(), base.g(), base.b());
    let stroke = Stroke::new(0.5, Color32::from_gray(30));

    painter.add(Shape::convex_polygon(face_z, shade(r, g, bl, -40), stroke));
    painter.add(Shape::convex_polygon(face_x, shade(r, g, bl, -10), stroke));
    painter.add(Shape::convex_polygon(top_face, shade(r, g, bl, 40), stroke));
}

fn draw_cursor(painter: &egui::Painter, cx: u16, cy: u16, cz: u16, rect: Rect, cam: &Camera, b: &Bounds) {
    let (x, y, z) = (cx as f32, cy as f32, cz as f32);
    let h = 0.5_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);
    let yellow = Color32::from_rgb(255, 230, 0);
    let stroke = Stroke::new(1.5, yellow);

    let top = vec![corner(-h, h, -h), corner(h, h, -h), corner(h, h, h), corner(-h, h, h)];
    painter.add(Shape::closed_line(top, stroke));
    let bot = vec![corner(-h, -h, -h), corner(h, -h, -h), corner(h, -h, h), corner(-h, -h, h)];
    painter.add(Shape::closed_line(bot, stroke));
    for (dx, dz) in [(-h, -h), (h, -h), (h, h), (-h, h)] {
        painter.line_segment([corner(dx, -h, dz), corner(dx, h, dz)], stroke);
    }
    painter.circle_filled(cam.to_screen(x, y, z, rect, b), 3.0, yellow);
}

// ---------------------------------------------------------------------------
// Shared state — mutated from JS via `load_trajectories_json`, read by the
// eframe App on every repaint.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ViewerState {
    trajectories: Vec<TrajectoryData>,
    iter_idx: usize,
    step_idx: usize,
    // Not wired up yet — occlusion toggle in the React sidebar isn't built (see README).
    #[allow(dead_code)]
    occlude_agent: bool,
}

struct ViewerApp {
    state: Rc<RefCell<ViewerState>>,
    camera: Camera,
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let drag_delta = ctx.input(|i| i.pointer.delta());
        let dragging = ctx.input(|i| i.pointer.primary_down());
        let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);
        if dragging {
            self.camera.yaw += drag_delta.x * 0.005;
            self.camera.pitch = (self.camera.pitch - drag_delta.y * 0.005).clamp(-1.4, 1.4);
        }
        if scroll_y != 0.0 {
            self.camera.zoom = (self.camera.zoom * (1.0 + scroll_y * 0.001)).clamp(0.2, 4.0);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let state = self.state.borrow();
            let Some(traj) = state.trajectories.get(state.iter_idx) else {
                ui.centered_and_justified(|ui| ui.label("No trajectory loaded"));
                return;
            };
            let rect = ui.available_rect_before_wrap();
            let painter = ui.painter_at(rect);
            let grid_size = traj.grid_size as f32;
            let bounds = self.camera.bounds(grid_size);

            let mut voxels: Vec<(u16, u16, u16)> =
                traj.episode.init_filled.iter().map(|c| (c[0], c[1], c[2])).collect();
            voxels.sort_by(|a, b| {
                depth_key(a.0, a.1, a.2, &self.camera)
                    .partial_cmp(&depth_key(b.0, b.1, b.2, &self.camera))
                    .unwrap()
            });
            for (x, y, z) in voxels {
                draw_voxel(&painter, x, y, z, rect, &self.camera, &bounds, Color32::from_rgb(90, 140, 220));
            }

            if let Some(step) = traj.episode.steps.get(state.step_idx) {
                draw_cursor(&painter, step.cursor_x, step.cursor_y, step.cursor_z, rect, &self.camera, &bounds);
            }
        });

        ctx.request_repaint();
    }
}

// ---------------------------------------------------------------------------
// wasm-bindgen entry points, called from web/src/panels/ReplayPanel.tsx
// ---------------------------------------------------------------------------

thread_local! {
    static SHARED_STATE: Rc<RefCell<ViewerState>> = Rc::new(RefCell::new(ViewerState::default()));
}

/// Mount the viewer onto `<canvas id="{canvas_id}">`. Call once from React on mount.
#[wasm_bindgen]
pub fn start(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let canvas_id = canvas_id.to_owned();
    let state = SHARED_STATE.with(|s| s.clone());

    wasm_bindgen_futures::spawn_local(async move {
        let runner = eframe::WebRunner::new();
        let web_options = eframe::WebOptions::default();
        let result = runner
            .start(
                web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.get_element_by_id(&canvas_id))
                    .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                    .expect("canvas element not found"),
                web_options,
                Box::new(move |_cc| Ok(Box::new(ViewerApp { state, camera: Camera::default() }))),
            )
            .await;
        if let Err(err) = result {
            web_sys::console::error_1(&format!("viewer-web failed to start: {err:?}").into());
        }
    });
    Ok(())
}

/// Replace the loaded trajectories. `json` is a `TrajectoryData[]` array —
/// the same shape the Tauri `load_trajectory` command returns.
#[wasm_bindgen]
pub fn load_trajectories_json(json: &str) -> Result<(), JsValue> {
    let trajectories: Vec<TrajectoryData> =
        serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    SHARED_STATE.with(|s| {
        let mut state = s.borrow_mut();
        state.trajectories = trajectories;
        state.iter_idx = 0;
        state.step_idx = 0;
    });
    Ok(())
}

/// Set which iteration/step is displayed (driven by the React sidebar's
/// scrubbers, not by input inside the canvas itself).
#[wasm_bindgen]
pub fn set_position(iter_idx: usize, step_idx: usize) {
    SHARED_STATE.with(|s| {
        let mut state = s.borrow_mut();
        state.iter_idx = iter_idx;
        state.step_idx = step_idx;
    });
}
