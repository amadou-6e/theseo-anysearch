//! Native policy explanation windows backed by one persistent Python scorer.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use eframe::egui::{self, Color32, RichText};
use serde_json::{json, Map, Value};

#[derive(Default)]
pub struct NativeExplainUi {
    bridge: Option<ExplanationBridge>,
    pub observation_open: bool,
    result_open: bool,
    observation: Map<String, Value>,
    fields: BTreeMap<String, FieldSchema>,
    result: Option<Value>,
    error: Option<String>,
    axis: usize,
    slice_index: usize,
    imported_observation: Option<String>,
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
        let mut ui = Self { observation_open, ..Self::default() };
        match ExplanationBridge::start(run, checkpoint) {
            Ok((bridge, ready)) => { ui.bridge = Some(bridge); ui.load_ready(&ready); }
            Err(error) => ui.error = Some(error),
        }
        ui
    }

    pub fn available(&self) -> bool { self.bridge.is_some() }

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
        if let Some(message) = self.error.clone() {
            ui.colored_label(Color32::LIGHT_RED, message);
            ui.separator();
        }
        ui.columns(2, |columns| {
            columns[0].heading("Observation editor");
            self.show_observation_editor(&mut columns[0]);
            columns[1].heading("Policy explanation");
            self.show_result(&mut columns[1]);
        });
    }

    fn show_observation_editor(&mut self, ui: &mut egui::Ui) {
        let mut explain = false;
        let mut changed = false;
        let mut import_path = None;
        ui.label("Edit normalized network inputs. Fictional observations are not environment-validated.");
        ui.horizontal(|ui| {
            if ui.button("Load fictional observation...").clicked() {
                import_path = rfd::FileDialog::new()
                    .add_filter("Observation", &["json", "npy", "npz", "pb", "tensor"])
                    .pick_file();
            }
            ui.small("Format detected automatically.");
        });
        if let Some(source) = &self.imported_observation {
            ui.colored_label(Color32::LIGHT_GREEN, format!("Loaded: {source}"));
        }
        ui.separator();
        egui::ScrollArea::vertical().id_salt("explanation_observation").show(ui, |ui| {
            changed |= self.show_local_grid(ui);
            ui.separator();
            changed |= self.show_scalar_fields(ui);
            ui.separator();
            explain = ui.button("Explain policy decision").clicked();
        });
        if let Some(path) = import_path {
            self.import_observation(&path);
        }
        if explain || changed {
            self.explain_observation();
        }
    }

    fn show_local_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let Some(values) = self.observation.get("local_grid").and_then(Value::as_array) else {
            ui.label("This policy has no local_grid field."); return false;
        };
        let side = (values.len() as f64).cbrt().round() as usize;
        if side.pow(3) != values.len() { ui.colored_label(Color32::LIGHT_RED, "local_grid is not cubic"); return false; }
        self.slice_index = self.slice_index.min(side.saturating_sub(1));
        ui.horizontal(|ui| {
            ui.label("Slice axis:");
            for (index, label) in ["X", "Y", "Z"].iter().enumerate() { ui.selectable_value(&mut self.axis, index, *label); }
            ui.add(egui::Slider::new(&mut self.slice_index, 0..=side - 1).text("index"));
        });
        let mut grid: Vec<f32> = values.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
        let mut changed = false;
        egui::Grid::new("observation_voxel_slice").spacing([3.0, 3.0]).show(ui, |ui| {
            for row in 0..side {
                for column in 0..side {
                    let index = match self.axis {
                        0 => self.slice_index * side * side + row * side + column,
                        1 => row * side * side + self.slice_index * side + column,
                        _ => row * side * side + column * side + self.slice_index,
                    };
                    let value = grid[index];
                    let shade = (value.clamp(0.0, 1.0) * 220.0) as u8;
                    if ui.add_sized([54.0, 28.0], egui::Button::new(format!("{value:.2}"))
                        .fill(Color32::from_rgb(shade, shade, shade))).clicked() {
                        grid[index] = if value < 0.16 { 1.0 / 3.0 } else if value < 0.5 { 2.0 / 3.0 }
                            else if value < 0.83 { 1.0 } else { 0.0 };
                        changed = true;
                    }
                }
                ui.end_row();
            }
        });
        self.observation.insert("local_grid".into(), json!(grid));
        ui.small("Click cells to cycle through 0, 1/3, 2/3, 1.");
        changed
    }

    fn show_scalar_fields(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let names: Vec<String> = self.observation.keys().filter(|name| name.as_str() != "local_grid").cloned().collect();
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
                for (index, score) in scores.iter().enumerate() {
                    let value = score.as_f64().unwrap_or(f64::NAN);
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{index:>2}"));
                        ui.add(egui::ProgressBar::new(value.clamp(0.0, 1.0) as f32).text(format!("{value:.5}")));
                    });
                }
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
