//! Native policy explanation windows backed by one persistent Python scorer.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};

use eframe::egui::{self, Color32, RichText};
use serde_json::{json, Map, Value};

#[derive(Default)]
pub struct NativeExplainUi {
    bridge: Option<ExplanationBridge>,
    startup: Option<Receiver<Result<(ExplanationBridge, Value), String>>>,
    pub observation_open: bool,
    result_open: bool,
    observation: Map<String, Value>,
    fields: BTreeMap<String, FieldSchema>,
    result: Option<Value>,
    error: Option<String>,
    axis: usize,
    slice_index: usize,
    imported_observation: Option<String>,
    camera_yaw: f32,
    camera_pitch: f32,
}

fn project_explanation_point(
    point: (f32, f32, f32), yaw: f32, pitch: f32,
    center: egui::Pos2, scale: f32,
) -> egui::Pos2 {
    let (x, y, z) = point;
    let xr = x * yaw.cos() - z * yaw.sin();
    let zr = x * yaw.sin() + z * yaw.cos();
    let yr = y * pitch.cos() - zr * pitch.sin();
    center + egui::vec2(xr * scale, -yr * scale)
}

fn draw_explanation_face(
    painter: &egui::Painter, points: [(f32, f32, f32); 4],
    yaw: f32, pitch: f32, center: egui::Pos2, scale: f32,
    base: Color32, alpha: u8, shade: i16,
) {
    let polygon = points.into_iter().map(|point|
        project_explanation_point(point, yaw, pitch, center, scale)).collect();
    let color = Color32::from_rgba_unmultiplied(
        (i16::from(base.r()) + shade).clamp(0, 255) as u8,
        (i16::from(base.g()) + shade).clamp(0, 255) as u8,
        (i16::from(base.b()) + shade).clamp(0, 255) as u8,
        alpha,
    );
    painter.add(egui::Shape::convex_polygon(
        polygon, color,
        egui::Stroke::new(0.7, Color32::from_rgba_unmultiplied(25, 28, 35, alpha)),
    ));
}

#[derive(Clone, Default)]
struct FieldSchema { low: Vec<f32>, high: Vec<f32> }

struct ExplanationBridge {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ExplanationBridge {
    fn start(run: &Path, checkpoint: &str) -> Result<(Self, Value), String> {
        let python = std::env::var("ANYSEARCH_PYTHON").unwrap_or_else(|_| "python".into());
        let mut child = Command::new(python)
            .args(["-u", "-m", "theseo_anysearch.rllib.explain.native_bridge", "--run"])
            .arg(run).args(["--checkpoint", checkpoint])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
            .spawn().map_err(|error| format!("failed to start explanation service: {error}"))?;
        let stdin = child.stdin.take().ok_or("explanation service has no stdin")?;
        let stdout = child.stdout.take().ok_or("explanation service has no stdout")?;
        let mut bridge = Self { _child: child, stdin, stdout: BufReader::new(stdout) };
        let ready = bridge.read_response()?;
        Ok((bridge, ready))
    }

    fn request(&mut self, request: Value) -> Result<Value, String> {
        serde_json::to_writer(&mut self.stdin, &request)
            .map_err(|error| format!("could not encode explanation request: {error}"))?;
        self.stdin.write_all(b"\n").map_err(|error| format!("could not send explanation request: {error}"))?;
        self.stdin.flush().map_err(|error| format!("could not flush explanation request: {error}"))?;
        self.read_response()
    }

    fn read_response(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        let bytes = self.stdout.read_line(&mut line)
            .map_err(|error| format!("could not read explanation response: {error}"))?;
        if bytes == 0 { return Err("explanation service exited without a response".into()); }
        let response: Value = serde_json::from_str(&line)
            .map_err(|error| format!("explanation service returned invalid JSON ({error}): {line}"))?;
        if !response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            let kind = response.get("error_type").and_then(Value::as_str).unwrap_or("Error");
            let message = response.get("error").and_then(Value::as_str)
                .unwrap_or("explanation request failed without a message");
            return Err(format!("{kind}: {message}"));
        }
        Ok(response)
    }
}

impl NativeExplainUi {
    pub fn start(run: &Path, checkpoint: &str, observation_open: bool) -> Self {
        let (sender, receiver) = mpsc::channel();
        let run = run.to_path_buf();
        let checkpoint = checkpoint.to_string();
        std::thread::spawn(move || {
            let _ = sender.send(ExplanationBridge::start(&run, &checkpoint));
        });
        Self {
            observation_open,
            startup: Some(receiver),
            camera_yaw: 45.0_f32.to_radians(),
            camera_pitch: 30.0_f32.to_radians(),
            ..Self::default()
        }
    }

    pub fn available(&self) -> bool { self.bridge.is_some() }

    fn poll_startup(&mut self) {
        let Some(receiver) = self.startup.take() else { return; };
        match receiver.try_recv() {
            Ok(Ok((bridge, ready))) => {
                self.bridge = Some(bridge);
                self.load_ready(&ready);
            }
            Ok(Err(error)) => self.error = Some(error),
            Err(TryRecvError::Empty) => self.startup = Some(receiver),
            Err(TryRecvError::Disconnected) => {
                self.error = Some("explanation service startup channel disconnected".into());
            }
        }
    }

    pub fn explain_trajectory(&mut self, trajectory: &Path, step: usize) {
        self.run_request(json!({"command": "explain_trajectory", "trajectory": trajectory, "step": step}));
    }

    fn explain_observation(&mut self) {
        self.run_request(json!({"command": "explain_observation", "observation": self.observation}));
    }


    fn import_observation(&mut self, path: &Path) {
        let request = json!({"command": "load_observation_file", "path": path});
        let Some(bridge) = self.bridge.as_mut() else {
            self.error = Some("no checkpoint-backed explanation service is configured".into());
            return;
        };
        match bridge.request(request) {
            Ok(response) => {
                let Some(observation) = response.get("observation").and_then(Value::as_object) else {
                    self.error = Some("observation import response omitted the observation object".into());
                    return;
                };
                self.observation = observation.clone();
                let format = response.get("format").and_then(Value::as_str).unwrap_or("unknown");
                self.imported_observation = Some(format!("{} ({format})", path.display()));
                self.error = None;
                self.explain_observation();
            }
            Err(error) => self.error = Some(error),
        }
    }
    fn run_request(&mut self, request: Value) {
        let Some(bridge) = self.bridge.as_mut() else {
            self.error = Some("no checkpoint-backed explanation service is configured".into());
            return;
        };
        match bridge.request(request) {
            Ok(response) => { self.result = response.get("report").cloned(); self.result_open = self.result.is_some(); self.error = None; }
            Err(error) => self.error = Some(error),
        }
    }

    fn load_ready(&mut self, ready: &Value) {
        self.observation = ready.get("observation").and_then(Value::as_object).cloned().unwrap_or_default();
        let Some(fields) = ready.get("fields").and_then(Value::as_object) else { return; };
        for (name, raw) in fields {
            let floats = |key: &str| raw.get(key).and_then(Value::as_array).map(|values| {
                values.iter().filter_map(|value| value.as_f64().map(|v| v as f32)).collect()
            }).unwrap_or_default();
            self.fields.insert(name.clone(), FieldSchema { low: floats("low"), high: floats("high") });
        }
    }

    pub fn show_embedded(&mut self, ui: &mut egui::Ui) {
        self.poll_startup();
        if self.startup.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Restoring policy checkpoint...");
            });
            ui.separator();
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
        if let Some(message) = self.error.clone() {
            ui.colored_label(Color32::LIGHT_RED, message);
            ui.separator();
        }
        let height = ui.available_height();
        // Winit can report the requested logical width while Windows clamps the
        // physical window to the monitor. Correct for that DPI mismatch so all
        // three panes remain inside the visible native window.
        let total_width = ui.available_width() / ui.ctx().pixels_per_point().max(1.0);
        let editor_width = (total_width * 0.20).clamp(180.0, 225.0);
        let result_width = (total_width * 0.20).clamp(190.0, 250.0);
        let geometry_width = (total_width - editor_width - result_width - 32.0).max(180.0);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                [editor_width, height].into(),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                ui.heading("Explain policy");
                self.show_observation_editor(ui);
                },
            );
            ui.separator();
            ui.allocate_ui_with_layout(
                [geometry_width, height].into(),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                ui.heading("Observation geometry");
                self.show_geometry_preview(ui);
                },
            );
            ui.separator();
            ui.allocate_ui_with_layout(
                [result_width, height].into(),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                ui.heading("Policy explanation");
                self.show_result(ui);
                },
            );
        });
    }

    fn show_observation_editor(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().slider_width = 90.0;
        let mut explain = false;
        let mut changed = false;
        let mut import_path = None;
        ui.label("Edit normalized network inputs. Fictional observations are not environment-validated.");
        if ui.button("Load fictional observation...").clicked() {
            import_path = rfd::FileDialog::new()
                .add_filter("Observation", &["json", "npy", "npz", "pb", "tensor"])
                .pick_file();
        }
        ui.small("The file type is detected automatically.");
        if let Some(source) = &self.imported_observation {
            ui.colored_label(Color32::LIGHT_GREEN, format!("Loaded: {source}"));
        }
        ui.separator();
        egui::ScrollArea::vertical().id_salt("explanation_observation").show(ui, |ui| {
            ui.label(RichText::new("Scalar observations").strong());
            changed |= self.show_scalar_fields(ui);
            ui.separator();
            ui.label(RichText::new("Active slice values").strong());
            changed |= self.show_local_grid(ui);
            ui.separator();
            explain = ui.button("Explain policy decision").clicked();
        });
        if let Some(path) = import_path {
            self.import_observation(&path);
        }
        if explain {
            self.explain_observation();
        }
        if changed {
            self.result = None;
        }
    }

    fn show_local_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let Some(values) = self.observation.get("local_grid").and_then(Value::as_array) else {
            ui.label("This policy has no local_grid field."); return false;
        };
        let side = (values.len() as f64).cbrt().round() as usize;
        if side == 0 || side.pow(3) != values.len() {
            ui.colored_label(Color32::LIGHT_RED, "local_grid is not cubic");
            return false;
        }
        self.slice_index = self.slice_index.min(side.saturating_sub(1));
        ui.horizontal(|ui| {
            ui.label("Slice axis:");
            for (index, label) in ["X", "Y", "Z"].iter().enumerate() { ui.selectable_value(&mut self.axis, index, *label); }
        });
        ui.add_sized(
            [140.0, 18.0],
            egui::Slider::new(&mut self.slice_index, 0..=side - 1).text("index"),
        );
        let mut grid: Vec<f32> = values.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
        let mut changed = false;
        egui::ScrollArea::horizontal().id_salt("observation_voxel_slice_scroll").show(ui, |ui| {
            egui::Grid::new("observation_voxel_slice").spacing([3.0, 3.0]).show(ui, |ui| {
            for row in 0..side {
                for column in 0..side {
                    let index = match self.axis {
                        0 => self.slice_index * side * side + row * side + column,
                        1 => row * side * side + self.slice_index * side + column,
                        _ => row * side * side + column * side + self.slice_index,
                    };
                    let value = &mut grid[index];
                    changed |= ui.add_sized(
                        [(180.0 / side as f32).clamp(36.0, 58.0), 27.0],
                        egui::DragValue::new(value)
                            .range(0.0..=1.0)
                            .speed(0.01)
                            .fixed_decimals(2),
                    ).changed();
                }
                ui.end_row();
            }
            });
        });
        self.observation.insert("local_grid".into(), json!(grid));
        ui.small("Click and type, or drag, to set an exact normalized value from 0 to 1.");
        changed
    }

    fn show_geometry_preview(&mut self, ui: &mut egui::Ui) {
        let Some(values) = self.observation.get("local_grid").and_then(Value::as_array) else {
            ui.label("This policy has no local_grid field.");
            return;
        };
        let side = (values.len() as f64).cbrt().round() as usize;
        if side == 0 || side.pow(3) != values.len() {
            ui.colored_label(Color32::LIGHT_RED, "local_grid is not cubic");
            return;
        }
        ui.label(format!(
            "{} slice {} / {} · non-selected slices remain translucent",
            ["X", "Y", "Z"][self.axis.min(2)],
            self.slice_index + 1,
            side,
        ));
        let size = ui.available_width().min(ui.available_height() - 48.0).max(240.0);
        let (response, painter) = ui.allocate_painter(egui::Vec2::splat(size), egui::Sense::drag());
        if response.dragged() {
            let delta = ui.ctx().input(|input| input.pointer.delta());
            self.camera_yaw += delta.x * 0.008;
            self.camera_pitch = (self.camera_pitch - delta.y * 0.008).clamp(-1.35, 1.35);
            ui.ctx().request_repaint();
        }
        let frame = response.rect.shrink(24.0);
        painter.rect_filled(frame, 12.0, Color32::from_rgb(10, 13, 18));
        let center_index = side / 2;
        let mut voxels = Vec::new();
        for x in 0..side { for y in 0..side { for z in 0..side {
            let value = values[x * side * side + y * side + z]
                .as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
            if value > 0.0 || (x == center_index && y == center_index && z == center_index) {
                voxels.push((x, y, z, value));
            }
        }}}
        let occupied: HashSet<(usize, usize, usize)> = voxels.iter()
            .map(|(x, y, z, _)| (*x, *y, *z)).collect();
        let view = (
            self.camera_yaw.sin() * self.camera_pitch.cos(),
            self.camera_pitch.sin(),
            self.camera_yaw.cos() * self.camera_pitch.cos(),
        );
        voxels.sort_by(|a, b| {
            let depth = |v: &(usize, usize, usize, f32)|
                v.0 as f32 * view.0 + v.1 as f32 * view.1 + v.2 as f32 * view.2;
            depth(a).partial_cmp(&depth(b)).unwrap_or(std::cmp::Ordering::Equal)
        });
        let cube_size = (frame.width() / (side as f32 * 1.9)).min(72.0);
        for (x, y, z, value) in voxels {
            let is_agent = x == center_index && y == center_index && z == center_index;
            let base = if is_agent || (value > 0.0 && value < 0.99) {
                Color32::from_rgb(70, 140, 210)
            } else {
                Color32::from_rgb(120, 120, 130)
            };
            let selected = match self.axis { 0 => x, 1 => y, _ => z } == self.slice_index;
            let alpha = if selected { 255 } else { 55 };
            let same_group_neighbor = |nx: isize, ny: isize, nz: isize| {
                if nx < 0 || ny < 0 || nz < 0 { return false; }
                let neighbor = (nx as usize, ny as usize, nz as usize);
                if !occupied.contains(&neighbor) { return false; }
                if !selected { return true; }
                match self.axis {
                    0 => neighbor.0 == self.slice_index,
                    1 => neighbor.1 == self.slice_index,
                    _ => neighbor.2 == self.slice_index,
                }
            };
            let (cx, cy, cz) = (
                x as f32 - center_index as f32,
                y as f32 - center_index as f32,
                z as f32 - center_index as f32,
            );
            let h = 0.5;
            let face = |axis: usize, sign: f32, shade: i16| {
                let neighbor = match axis {
                    0 => (x as isize + sign as isize, y as isize, z as isize),
                    1 => (x as isize, y as isize + sign as isize, z as isize),
                    _ => (x as isize, y as isize, z as isize + sign as isize),
                };
                if same_group_neighbor(neighbor.0, neighbor.1, neighbor.2) { return; }
                let points = match axis {
                    0 => [(cx+sign*h,cy-h,cz-h),(cx+sign*h,cy-h,cz+h),(cx+sign*h,cy+h,cz+h),(cx+sign*h,cy+h,cz-h)],
                    1 => [(cx-h,cy+sign*h,cz-h),(cx+h,cy+sign*h,cz-h),(cx+h,cy+sign*h,cz+h),(cx-h,cy+sign*h,cz+h)],
                    _ => [(cx-h,cy-h,cz+sign*h),(cx+h,cy-h,cz+sign*h),(cx+h,cy+h,cz+sign*h),(cx-h,cy+h,cz+sign*h)],
                };
                draw_explanation_face(&painter, points, self.camera_yaw, self.camera_pitch,
                    frame.center(), cube_size, base, alpha, shade);
            };
            face(0, if view.0 >= 0.0 { 1.0 } else { -1.0 }, -10);
            face(1, if view.1 >= 0.0 { 1.0 } else { -1.0 }, 35);
            face(2, if view.2 >= 0.0 { 1.0 } else { -1.0 }, -40);
        }
        painter.text(frame.right_bottom(), egui::Align2::RIGHT_BOTTOM,
            "Drag to rotate", egui::FontId::proportional(11.0), Color32::from_gray(100));
    }

    fn show_scalar_fields(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let names: Vec<String> = self.observation.keys().filter(|name| name.as_str() != "local_grid").cloned().collect();
        if names.is_empty() {
            ui.colored_label(Color32::YELLOW, "This policy exposes no scalar observation fields.");
            return false;
        }
        for name in names {
            let Some(values) = self.observation.get(&name).and_then(Value::as_array) else { continue; };
            let mut edited: Vec<f32> = values.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
            let bounds = self.fields.get(&name).cloned().unwrap_or_default();
            ui.label(RichText::new(&name).strong());
            for (index, value) in edited.iter_mut().enumerate() {
                let low = bounds.low.get(index).or(bounds.low.first()).copied().unwrap_or(-1.0);
                let high = bounds.high.get(index).or(bounds.high.first()).copied().unwrap_or(1.0);
                changed |= ui.add(egui::Slider::new(value, low..=high).text(format!("[{index}]"))).changed();
            }
            self.observation.insert(name, json!(edited));
        }
        changed
    }

    fn show_result(&mut self, ui: &mut egui::Ui) {
        let result = self.result.clone();
        egui::ScrollArea::vertical().id_salt("explanation_result").show(ui, |ui| {
            let Some(step) = result.as_ref().and_then(|report| report.get("steps"))
                .and_then(Value::as_array).and_then(|steps| steps.first()) else {
                ui.label("Explain a replay step or edited observation to see policy scores.");
                return;
            };
            let action = step.get("chosen_action").and_then(Value::as_i64).unwrap_or(-1);
            let direction = step.get("chosen_direction").cloned().unwrap_or(Value::Null);
            let margin = step.get("score_margin").and_then(Value::as_f64).unwrap_or(f64::NAN);
            ui.heading(format!("Action {action}  {direction}"));
            ui.label(format!("Margin over best safe action: {margin:.6}"));
            ui.separator();
            ui.label(RichText::new("Action scores").strong());
            if let Some(scores) = step.get("action_scores").and_then(Value::as_array) {
                let finite: Vec<f64> = scores.iter().filter_map(Value::as_f64)
                    .filter(|value| value.is_finite()).collect();
                let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
                let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let span = (maximum - minimum).max(f64::EPSILON);
                for (index, score) in scores.iter().enumerate() {
                    let value = score.as_f64().unwrap_or(f64::NAN);
                    let normalized = ((value - minimum) / span).clamp(0.0, 1.0) as f32;
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{index:>2}"));
                        ui.add(egui::ProgressBar::new(normalized).text(format!("{value:.5}")));
                    });
                }
                ui.small("Bars are min-max normalized across this decision; labels show raw policy scores.");
            }
            ui.separator();
            ui.label(RichText::new("Grouped attribution").strong());
            if let Some(groups) = step.get("group_attributions").and_then(Value::as_object) {
                for (name, value) in groups {
                    ui.label(format!("{name}: {:.6}", value.as_f64().unwrap_or(f64::NAN)));
                }
            }
        });
    }

}
