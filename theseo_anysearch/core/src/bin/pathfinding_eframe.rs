use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Slider, Stroke, Vec2};
use theseo_core::environments::Environment as _;
use theseo_core::surface::{AgentState, SurfaceAction, SurfaceEnv};
use theseo_core::voxel::world::ingest::{parse_ascii_stl, voxelize_mesh};

#[derive(Clone)]
struct EpisodeStep {
    step: u32,
    agents: Vec<AgentState>,
    new_paints: Vec<((u16, u16, u16), usize)>,
}

#[derive(Clone)]
struct EpisodeTrace {
    epoch: u32,
    geometry_coords: Vec<(u16, u16, u16)>,
    surface_coords: Vec<(u16, u16, u16)>,
    steps: Vec<EpisodeStep>,
    reached: usize,
    total_agents: usize,
}

fn run_episode_trace(
    stl_ascii: &str,
    origin: (u16, u16, u16),
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
    epoch: u32,
) -> Result<EpisodeTrace, String> {
    let mesh = parse_ascii_stl(stl_ascii).map_err(|e| format!("{e:?}"))?;
    let placements = voxelize_mesh(&mesh, origin, scale);
    let filled = placements.iter().map(|p| p.coord).collect::<Vec<_>>();

    let mut env = SurfaceEnv::from_filled_surface(&filled, agent_count, max_steps);
    let mut obs = env.reset(seed);
    let mut steps = 0u32;
    let mut seen = HashSet::new();
    let mut trace_steps = Vec::new();

    let mut first = Vec::new();
    for (coord, owner) in env.trail_owner() {
        seen.insert(*coord);
        first.push((*coord, *owner));
    }
    trace_steps.push(EpisodeStep {
        step: 0,
        agents: obs.agents.clone(),
        new_paints: first,
    });

    while steps < max_steps && obs.reached_agents < obs.total_agents {
        let sr = env.step(SurfaceAction::StepAll);
        obs = sr.observation;
        steps += 1;
        let mut new_paints = Vec::new();
        for (coord, owner) in env.trail_owner() {
            if seen.insert(*coord) {
                new_paints.push((*coord, *owner));
            }
        }
        trace_steps.push(EpisodeStep {
            step: steps,
            agents: obs.agents.clone(),
            new_paints,
        });
        if sr.done {
            break;
        }
    }

    Ok(EpisodeTrace {
        epoch,
        geometry_coords: filled,
        surface_coords: env.surface_coords(),
        steps: trace_steps,
        reached: obs.reached_agents,
        total_agents: obs.total_agents,
    })
}

fn project_iso(coord: (u16, u16, u16)) -> (f32, f32) {
    let x = coord.0 as f32;
    let y = coord.1 as f32;
    let z = coord.2 as f32;
    let yaw = 35.0_f32.to_radians();
    let pitch = 28.0_f32.to_radians();
    let xr = x * yaw.cos() - z * yaw.sin();
    let zr = x * yaw.sin() + z * yaw.cos();
    let yr = y * pitch.cos() - zr * pitch.sin();
    let zr2 = y * pitch.sin() + zr * pitch.cos();
    (xr, yr - zr2 * 0.15)
}

struct Bounds {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

fn build_bounds(trace: &EpisodeTrace) -> Bounds {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for c in &trace.geometry_coords {
        let (x, y) = project_iso(*c);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    for c in &trace.surface_coords {
        let (x, y) = project_iso(*c);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    for s in &trace.steps {
        for a in &s.agents {
            for c in [a.current, a.target] {
                let (x, y) = project_iso(c);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    Bounds {
        min_x,
        max_x,
        min_y,
        max_y,
    }
}

fn map_to_screen(c: (u16, u16, u16), rect: Rect, b: &Bounds) -> Pos2 {
    let (x, y) = project_iso(c);
    let sx = (x - b.min_x) / (b.max_x - b.min_x).max(1.0);
    let sy = (y - b.min_y) / (b.max_y - b.min_y).max(1.0);
    Pos2::new(
        rect.left() + sx * rect.width(),
        rect.top() + sy * rect.height(),
    )
}

struct PathfindingViewerApp {
    traces: Vec<EpisodeTrace>,
    epoch_idx: usize,
    step_idx: usize,
    playing: bool,
}

impl PathfindingViewerApp {
    fn new(traces: Vec<EpisodeTrace>) -> Self {
        Self {
            traces,
            epoch_idx: 0,
            step_idx: 0,
            playing: true,
        }
    }
}

impl eframe::App for PathfindingViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let colors = [
            Color32::from_rgb(230, 25, 75),
            Color32::from_rgb(60, 180, 75),
            Color32::from_rgb(0, 130, 200),
            Color32::from_rgb(245, 130, 48),
            Color32::from_rgb(145, 30, 180),
        ];

        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Pathfinding Viewer");
            ui.separator();
            if !self.traces.is_empty() {
                let max_epoch_idx = self.traces.len().saturating_sub(1);
                ui.add(Slider::new(&mut self.epoch_idx, 0..=max_epoch_idx).text("Epoch index"));
                let trace = &self.traces[self.epoch_idx];
                ui.label(format!("Epoch: {}", trace.epoch));
                ui.label(format!("Reached: {}/{}", trace.reached, trace.total_agents));
                let max_step_idx = trace.steps.len().saturating_sub(1);
                if self.step_idx > max_step_idx {
                    self.step_idx = max_step_idx;
                }
                ui.add(Slider::new(&mut self.step_idx, 0..=max_step_idx).text("Step"));
                ui.checkbox(&mut self.playing, "Auto-play");
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
            painter.rect_filled(resp.rect, 0.0, Color32::from_rgb(15, 18, 24));

            if self.traces.is_empty() {
                painter.text(
                    resp.rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "No traces",
                    egui::TextStyle::Heading.resolve(ui.style()),
                    Color32::WHITE,
                );
                return;
            }

            let trace = &self.traces[self.epoch_idx];
            let b = build_bounds(trace);
            let mut painted: HashMap<(u16, u16, u16), usize> = HashMap::new();
            for i in 0..=self.step_idx.min(trace.steps.len().saturating_sub(1)) {
                for (coord, owner) in &trace.steps[i].new_paints {
                    painted.insert(*coord, *owner);
                }
            }

            for c in &trace.geometry_coords {
                let p = map_to_screen(*c, resp.rect.shrink(20.0), &b);
                let r = Rect::from_center_size(p, Vec2::splat(3.0));
                painter.rect_filled(r, 0.0, Color32::from_rgb(95, 95, 105));
            }

            for c in &trace.surface_coords {
                let p = map_to_screen(*c, resp.rect.shrink(20.0), &b);
                let r = Rect::from_center_size(p, Vec2::splat(2.0));
                painter.rect_filled(r, 0.0, Color32::from_gray(90));
            }

            for (coord, owner) in painted {
                let p = map_to_screen(coord, resp.rect.shrink(20.0), &b);
                let r = Rect::from_center_size(p, Vec2::splat(4.0));
                painter.rect_filled(r, 0.0, colors[owner % colors.len()]);
            }

            let s = &trace.steps[self.step_idx.min(trace.steps.len().saturating_sub(1))];
            for (idx, a) in s.agents.iter().enumerate() {
                let t = map_to_screen(a.target, resp.rect.shrink(20.0), &b);
                painter.circle_stroke(t, 3.5, Stroke::new(1.5, Color32::WHITE));
                let c = map_to_screen(a.current, resp.rect.shrink(20.0), &b);
                painter.circle_filled(c, 5.0, colors[idx % colors.len()]);
            }
        });

        if self.playing && !self.traces.is_empty() {
            let max_step = self.traces[self.epoch_idx].steps.len().saturating_sub(1);
            if self.step_idx < max_step {
                self.step_idx += 1;
            } else {
                self.epoch_idx = (self.epoch_idx + 1) % self.traces.len();
                self.step_idx = 0;
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(60));
        }
    }
}

fn main() -> eframe::Result<()> {
    let mut args = std::env::args().skip(1);
    let stl_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../../python/cli/examples/complex_slotted_disk.stl"));
    let epochs: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let video_every: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let stl = std::fs::read_to_string(&stl_path)
        .map_err(|e| eframe::Error::AppCreation(format!("read stl failed: {e}").into()))?;

    let mut traces = Vec::new();
    for epoch in 1..=epochs {
        if epoch % video_every != 0 {
            continue;
        }
        let trace = run_episode_trace(
            &stl,
            (260, 260, 260),
            40.0,
            5,
            400,
            3030 + epoch as u64,
            epoch,
        )
        .map_err(|e| eframe::Error::AppCreation(e.into()))?;
        traces.push(trace);
    }

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Pathfinding Eframe Inspector",
        options,
        Box::new(move |_cc| Ok(Box::new(PathfindingViewerApp::new(traces)))),
    )
}
