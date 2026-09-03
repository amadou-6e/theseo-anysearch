/// Garden Show-Data Viewer
///
/// Loads a `.gobs` binary file written by `anysearch garden show-data` and
/// displays observation samples with optional autoencoder reconstructions.
///
/// Binary format (little-endian):
///   4 bytes  magic  "GOBS"
///   4 bytes  u32    n_samples
///   1 byte   u8     n  (grid dimension per axis, e.g. 5 for box_radius=2)
///   1 byte   u8     flags  (bit 0 = has_recon)
///   2 bytes  pad
///   n_samples * n^3 * 4 bytes  f32  inputs
///   [if has_recon] n_samples * n^3 * 4 bytes  f32  reconstructions
///
/// Controls:
///   ← / →   prev / next sample
///   I/R/D   switch view mode: Input / Recon / Diff
///   Drag     orbit camera
///   Scroll   zoom
use std::path::PathBuf;

use eframe::egui::{self, Color32, Key, Pos2, Rect, ScrollArea, Sense, Shape, Slider, Stroke, Vec2};

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

struct GobsData {
    n_samples: usize,
    n: usize,
    has_recon: bool,
    data: Vec<f32>,
    recon: Vec<f32>,  // empty if !has_recon
}

impl GobsData {
    fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{e}"))?;
        if bytes.len() < 12 || &bytes[0..4] != b"GOBS" {
            return Err("not a .gobs file".into());
        }
        let n_samples = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let n = bytes[8] as usize;
        let flags = bytes[9];
        let has_recon = flags & 0x01 != 0;
        let sample_floats = n * n * n;
        let data_bytes = n_samples * sample_floats * 4;
        let expected = 12 + data_bytes + if has_recon { data_bytes } else { 0 };
        if bytes.len() < expected {
            return Err(format!("truncated: got {} bytes, need {expected}", bytes.len()));
        }
        let parse_f32 = |slice: &[u8]| -> Vec<f32> {
            slice.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let data = parse_f32(&bytes[12..12 + data_bytes]);
        let recon = if has_recon {
            parse_f32(&bytes[12 + data_bytes..12 + data_bytes * 2])
        } else {
            vec![]
        };
        Ok(Self { n_samples, n, has_recon, data, recon })
    }

    fn sample(&self, idx: usize) -> &[f32] {
        let sz = self.n * self.n * self.n;
        &self.data[idx * sz..(idx + 1) * sz]
    }

    fn recon_sample(&self, idx: usize) -> Option<&[f32]> {
        if !self.has_recon { return None; }
        let sz = self.n * self.n * self.n;
        Some(&self.recon[idx * sz..(idx + 1) * sz])
    }

    fn cell(&self, sample: &[f32], x: usize, y: usize, z: usize) -> f32 {
        sample[x * self.n * self.n + y * self.n + z]
    }

    fn fill_pct(&self, idx: usize) -> f32 {
        let s = self.sample(idx);
        let filled = s.iter().filter(|&&v| v > 0.5).count();
        100.0 * filled as f32 / s.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

impl Camera {
    fn new() -> Self {
        Self { yaw: 45.0_f32.to_radians(), pitch: 30.0_f32.to_radians(), zoom: 1.0 }
    }

    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let xr = x * self.yaw.cos() - z * self.yaw.sin();
        let zr = x * self.yaw.sin() + z * self.yaw.cos();
        let yr = y * self.pitch.cos() - zr * self.pitch.sin();
        let zr2 = y * self.pitch.sin() + zr * self.pitch.cos();
        (xr, -(yr - zr2 * 0.05))
    }

    fn bounds(&self, grid: f32) -> Bounds {
        let lo = 0.5_f32;
        let hi = grid + 0.5;
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

struct Bounds { min_x: f32, max_x: f32, min_y: f32, max_y: f32 }

fn depth_key(x: usize, y: usize, z: usize, cam: &Camera) -> f32 {
    let (sy, cy) = (cam.yaw.sin(), cam.yaw.cos());
    let (sp, cp) = (cam.pitch.sin(), cam.pitch.cos());
    x as f32 * sy * cp + y as f32 * sp + z as f32 * cy * cp
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
    )
}

fn draw_voxel(
    painter: &egui::Painter,
    cx: usize, cy: usize, cz: usize,
    rect: Rect, cam: &Camera, b: &Bounds,
    base: Color32,
) {
    let (x, y, z) = (cx as f32, cy as f32, cz as f32);
    let h = 0.5_f32;
    let corner = |dx: f32, dy: f32, dz: f32| cam.to_screen(x + dx, y + dy, z + dz, rect, b);

    let hx = if cam.yaw.sin() > 0.0 { h } else { -h };
    let hy = if cam.pitch.sin() > 0.0 { h } else { -h };
    let hz = if cam.yaw.cos() > 0.0 { h } else { -h };

    let shade = |r: u8, g: u8, bl: u8, amount: i32| -> Color32 {
        Color32::from_rgb(
            (r as i32 + amount).clamp(0, 255) as u8,
            (g as i32 + amount).clamp(0, 255) as u8,
            (bl as i32 + amount).clamp(0, 255) as u8,
        )
    };
    let (r, g, bl) = (base.r(), base.g(), base.b());
    let stroke = Stroke::new(0.5, Color32::from_gray(30));

    painter.add(Shape::convex_polygon(
        vec![corner(-h,hy,-h), corner(h,hy,-h), corner(h,hy,h), corner(-h,hy,h)],
        shade(r, g, bl, 40), stroke,
    ));
    painter.add(Shape::convex_polygon(
        vec![corner(hx,-h,-h), corner(hx,-h,h), corner(hx,h,h), corner(hx,h,-h)],
        shade(r, g, bl, -10), stroke,
    ));
    painter.add(Shape::convex_polygon(
        vec![corner(-h,-h,hz), corner(h,-h,hz), corner(h,h,hz), corner(-h,h,hz)],
        shade(r, g, bl, -40), stroke,
    ));
}

fn draw_wireframe_box(painter: &egui::Painter, rect: Rect, cam: &Camera, b: &Bounds, n: f32) {
    let stroke = Stroke::new(1.0, Color32::from_rgba_premultiplied(80, 80, 80, 60));
    let lo = 0.5_f32;
    let hi = n + 0.5;
    let corners = [
        (lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo),
        (lo, lo, hi), (hi, lo, hi), (hi, hi, hi), (lo, hi, hi),
    ];
    let edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ];
    let pts: Vec<Pos2> = corners.iter()
        .map(|&(x,y,z)| cam.to_screen(x, y, z, rect, b))
        .collect();
    for (a, b_idx) in edges {
        painter.line_segment([pts[a], pts[b_idx]], stroke);
    }
}

// ---------------------------------------------------------------------------
// View mode
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
enum ViewMode { Input, Recon, Diff }

// ---------------------------------------------------------------------------
// 2-D slice panel  (3 columns when reconstruction is available)
// ---------------------------------------------------------------------------

fn draw_slice_panel(ui: &mut egui::Ui, data: &GobsData, idx: usize, mode: ViewMode) {
    let inp = data.sample(idx);
    let rec = data.recon_sample(idx);
    let n = data.n;
    let mid = n / 2;
    let cell_size = 13.0_f32;

    // Color functions
    let input_col = |v: f32| -> Color32 {
        if v > 0.5 { Color32::from_gray(40) } else { Color32::from_gray(220) }
    };
    let recon_col = |v: f32| -> Color32 {
        // white → deep blue gradient
        lerp_color(Color32::from_rgb(240, 240, 255), Color32::from_rgb(20, 60, 160), v)
    };
    let diff_col = |inp_v: f32, rec_v: f32| -> Color32 {
        let i = if inp_v > 0.5 { 1.0_f32 } else { 0.0 };
        let r = if rec_v > 0.5 { 1.0_f32 } else { 0.0 };
        match (i > 0.5, r > 0.5) {
            (true,  true)  => Color32::from_rgb(80, 170, 80),    // true positive — green
            (true,  false) => Color32::from_rgb(220, 100, 30),   // false negative — orange
            (false, true)  => Color32::from_rgb(60, 120, 210),   // false positive — blue
            (false, false) => Color32::from_gray(230),           // true negative — light
        }
    };

    // Draw one n×n grid, returning the response for center-highlight
    let draw_grid = |ui: &mut egui::Ui, get_col: &dyn Fn(usize, usize) -> Color32| {
        let (resp, painter) = ui.allocate_painter(
            Vec2::new(n as f32 * cell_size, n as f32 * cell_size),
            Sense::hover(),
        );
        let tl = resp.rect.min;
        for i in 0..n {
            for j in 0..n {
                let x = tl.x + j as f32 * cell_size;
                let y = tl.y + i as f32 * cell_size;
                painter.rect_filled(
                    Rect::from_min_size(Pos2::new(x, y), Vec2::splat(cell_size - 1.0)),
                    0.0,
                    get_col(i, j),
                );
            }
        }
        // Centre cell highlight
        let cx = tl.x + mid as f32 * cell_size;
        let cy = tl.y + mid as f32 * cell_size;
        painter.rect_stroke(
            Rect::from_min_size(Pos2::new(cx, cy), Vec2::splat(cell_size - 1.0)),
            0.0,
            Stroke::new(1.5, Color32::from_rgb(0, 160, 220)),
            egui::StrokeKind::Outside,
        );
    };

    // Column labels + per-axis row labels
    let slices: &[(&str, Box<dyn Fn(usize, usize) -> (f32, f32)>)] = &[
        ("XY z=mid", Box::new(|i, j| (data.cell(inp, i, j, mid), rec.map_or(0.0, |r| data.cell(r, i, j, mid))))),
        ("XZ y=mid", Box::new(|i, j| (data.cell(inp, i, mid, j), rec.map_or(0.0, |r| data.cell(r, i, mid, j))))),
        ("YZ x=mid", Box::new(|i, j| (data.cell(inp, mid, i, j), rec.map_or(0.0, |r| data.cell(r, mid, i, j))))),
    ];

    if rec.is_none() || mode == ViewMode::Input {
        // Single column: just show input
        ui.vertical(|ui| {
            for (label, get) in slices {
                ui.label(egui::RichText::new(*label).small().weak());
                draw_grid(ui, &|i, j| input_col(get(i, j).0));
                ui.add_space(4.0);
            }
        });
        return;
    }

    // Three-column layout: Input | Recon | Diff
    let col_headers = ["Input", "Recon", "Diff"];
    let col_width = n as f32 * cell_size + 6.0;

    ui.vertical(|ui| {
        // Column headers
        ui.horizontal(|ui| {
            for h in col_headers {
                ui.allocate_ui(Vec2::new(col_width, 16.0), |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.label(egui::RichText::new(h).small().strong());
                    });
                });
            }
        });

        for (label, get) in slices {
            ui.label(egui::RichText::new(*label).small().weak());
            ui.horizontal(|ui| {
                // Input column
                draw_grid(ui, &|i, j| input_col(get(i, j).0));
                ui.add_space(3.0);
                // Recon column
                draw_grid(ui, &|i, j| recon_col(get(i, j).1));
                ui.add_space(3.0);
                // Diff column
                draw_grid(ui, &|i, j| {
                    let (iv, rv) = get(i, j);
                    diff_col(iv, rv)
                });
            });
            ui.add_space(4.0);
        }
    });
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct GardenShowApp {
    data: GobsData,
    path: PathBuf,
    fills: Vec<f32>,

    sample_idx: usize,
    cam: Camera,

    filter_min: f32,
    filter_max: f32,
    visible: Vec<usize>,
    vis_pos: usize,
    scroll_to_selected: bool,

    view_mode: ViewMode,
}

impl GardenShowApp {
    fn new(data: GobsData, path: PathBuf) -> Self {
        let fills: Vec<f32> = (0..data.n_samples).map(|i| data.fill_pct(i)).collect();
        let filter_min = if fills.iter().any(|&f| f > 0.0) { 0.1 } else { 0.0 };
        let visible: Vec<usize> = (0..fills.len()).filter(|&i| fills[i] >= filter_min).collect();
        let sample_idx = visible.first().copied().unwrap_or(0);
        Self {
            data, path, fills,
            sample_idx,
            cam: Camera::new(),
            filter_min, filter_max: 100.0,
            visible,
            vis_pos: 0,
            scroll_to_selected: false,
            view_mode: ViewMode::Input,
        }
    }

    fn rebuild_visible(&mut self) {
        self.visible = (0..self.data.n_samples)
            .filter(|&i| { let f = self.fills[i]; f >= self.filter_min && f <= self.filter_max })
            .collect();
        self.vis_pos = self.vis_pos.min(self.visible.len().saturating_sub(1));
        if let Some(&idx) = self.visible.get(self.vis_pos) {
            self.sample_idx = idx;
        }
        self.scroll_to_selected = true;
    }

    fn go_prev(&mut self) {
        if self.vis_pos > 0 {
            self.vis_pos -= 1;
            self.sample_idx = self.visible[self.vis_pos];
            self.scroll_to_selected = true;
        }
    }

    fn go_next(&mut self) {
        if self.vis_pos + 1 < self.visible.len() {
            self.vis_pos += 1;
            self.sample_idx = self.visible[self.vis_pos];
            self.scroll_to_selected = true;
        }
    }
}

impl eframe::App for GardenShowApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keyboard navigation
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft))  { self.go_prev(); }
        if ctx.input(|i| i.key_pressed(Key::ArrowRight)) { self.go_next(); }
        if self.data.has_recon {
            if ctx.input(|i| i.key_pressed(Key::I)) { self.view_mode = ViewMode::Input; }
            if ctx.input(|i| i.key_pressed(Key::R)) { self.view_mode = ViewMode::Recon; }
            if ctx.input(|i| i.key_pressed(Key::D)) { self.view_mode = ViewMode::Diff; }
        }

        // Top bar
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let file_name = self.path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                ui.label(egui::RichText::new(file_name).strong());
                ui.separator();
                ui.label(format!("{} samples", self.data.n_samples));
                ui.separator();

                let fill = if self.visible.is_empty() { 0.0 } else { self.fills[self.sample_idx] };
                ui.label(format!("sample {} / {}   fill {:.1}%", self.vis_pos + 1, self.visible.len(), fill));

                ui.separator();
                ui.label("filter:");
                let old_min = self.filter_min;
                let old_max = self.filter_max;
                ui.add(Slider::new(&mut self.filter_min, 0.0..=100.0).text("min%").integer());
                ui.add(Slider::new(&mut self.filter_max, 0.0..=100.0).text("max%").integer());
                if (self.filter_min - old_min).abs() > 0.1 || (self.filter_max - old_max).abs() > 0.1 {
                    self.rebuild_visible();
                }

                ui.separator();
                if ui.button("◄").clicked() { self.go_prev(); }
                if ui.button("►").clicked() { self.go_next(); }

                // View mode buttons (only when reconstruction is available)
                if self.data.has_recon {
                    ui.separator();
                    ui.label("view:");
                    if ui.selectable_label(self.view_mode == ViewMode::Input, "Input [I]").clicked() {
                        self.view_mode = ViewMode::Input;
                    }
                    if ui.selectable_label(self.view_mode == ViewMode::Recon, "Recon [R]").clicked() {
                        self.view_mode = ViewMode::Recon;
                    }
                    if ui.selectable_label(self.view_mode == ViewMode::Diff, "Diff [D]").clicked() {
                        self.view_mode = ViewMode::Diff;
                    }
                }
            });
        });

        // Right panel: 2D slices (wider when showing 3 columns)
        let panel_width = if self.data.has_recon { 360.0_f32 } else { 120.0_f32 };
        egui::SidePanel::right("slices").exact_width(panel_width).show(ctx, |ui| {
            ui.heading("Slices");
            ui.separator();
            if !self.visible.is_empty() {
                draw_slice_panel(ui, &self.data, self.sample_idx, self.view_mode);
            }
        });

        // Left panel: sample list
        let do_scroll = self.scroll_to_selected;
        self.scroll_to_selected = false;
        egui::SidePanel::left("sample_list").min_width(110.0).max_width(160.0).show(ctx, |ui| {
            ui.heading("Samples");
            ui.separator();
            ScrollArea::vertical().show(ui, |ui| {
                for (pos, &idx) in self.visible.iter().enumerate() {
                    let selected = pos == self.vis_pos;
                    let label = format!("{idx}  {:.0}%", self.fills[idx]);
                    let resp = ui.selectable_label(selected, &label);
                    if resp.clicked() {
                        self.vis_pos = pos;
                        self.sample_idx = idx;
                    }
                    if selected && do_scroll {
                        resp.scroll_to_me(None);
                    }
                }
            });
        });

        // Main: 3D voxel view
        egui::CentralPanel::default().show(ctx, |ui| {
            let mode_hint = if self.data.has_recon {
                "drag to orbit  •  scroll to zoom  •  I/R/D = Input/Recon/Diff"
            } else {
                "drag to orbit  •  scroll to zoom"
            };
            ui.label(egui::RichText::new(mode_hint).small().weak());

            let avail = ui.available_size();
            let (resp, painter) = ui.allocate_painter(avail, Sense::click_and_drag());
            let rect = resp.rect;

            if resp.dragged() {
                let delta = resp.drag_delta();
                self.cam.yaw   += delta.x * 0.008;
                self.cam.pitch += delta.y * 0.008;
                self.cam.pitch = self.cam.pitch.clamp(-1.4, 1.4);
            }
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                self.cam.zoom = (self.cam.zoom * (1.0 + scroll * 0.001)).clamp(0.2, 5.0);
            }

            if self.visible.is_empty() {
                painter.text(rect.center(), egui::Align2::CENTER_CENTER,
                    "No samples match filter", egui::FontId::default(), Color32::GRAY);
                return;
            }

            let n = self.data.n;
            let b = self.cam.bounds(n as f32);
            painter.rect_filled(rect, 0.0, Color32::from_gray(245));
            draw_wireframe_box(&painter, rect, &self.cam, &b, n as f32);

            let inp = self.data.sample(self.sample_idx);
            let rec = self.data.recon_sample(self.sample_idx);

            // Build voxel list depending on view mode
            let mut voxels: Vec<(usize, usize, usize, Color32, f32)> = Vec::new();
            for x in 0..n {
                for y in 0..n {
                    for z in 0..n {
                        let iv = self.data.cell(inp, x, y, z);
                        let rv = rec.map_or(0.0, |r| self.data.cell(r, x, y, z));
                        let dk = depth_key(x, y, z, &self.cam);

                        let col = match self.view_mode {
                            ViewMode::Input => {
                                if iv > 0.5 { Some(Color32::from_rgb(110, 120, 130)) } else { None }
                            }
                            ViewMode::Recon => {
                                if rv > 0.5 {
                                    // Colour by confidence: dark blue = high prob
                                    let c = lerp_color(
                                        Color32::from_rgb(140, 180, 220),
                                        Color32::from_rgb(20, 60, 140),
                                        rv,
                                    );
                                    Some(c)
                                } else { None }
                            }
                            ViewMode::Diff => {
                                // Only show disagreements
                                let i_filled = iv > 0.5;
                                let r_filled = rv > 0.5;
                                match (i_filled, r_filled) {
                                    (true, false) => Some(Color32::from_rgb(220, 100, 30)), // missed
                                    (false, true) => Some(Color32::from_rgb(60, 120, 210)), // phantom
                                    _ => None,
                                }
                            }
                        };
                        if let Some(c) = col {
                            voxels.push((x, y, z, c, dk));
                        }
                    }
                }
            }

            voxels.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
            for (x, y, z, col, _) in voxels {
                draw_voxel(&painter, x, y, z, rect, &self.cam, &b, col);
            }

            // Centre marker
            let mid = n / 2;
            let cp = self.cam.to_screen(mid as f32, mid as f32, mid as f32, rect, &b);
            painter.circle_stroke(cp, 5.0, Stroke::new(1.5, Color32::from_rgba_premultiplied(0, 180, 220, 120)));
        });
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: garden-show-data <file.gobs>");
        std::process::exit(1);
    }
    let path = PathBuf::from(&args[1]);
    let data = GobsData::load(&path).unwrap_or_else(|e| {
        eprintln!("Error loading {}: {e}", path.display());
        std::process::exit(1);
    });

    let recon_note = if data.has_recon { "  +recon" } else { "" };
    let title = format!("Garden Show-Data — {} samples  n={}{}", data.n_samples, data.n, recon_note);
    let app = GardenShowApp::new(data, path);

    eframe::run_native(
        &title,
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 720.0])
                .with_title(&title),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(app))),
    ).unwrap();
}
