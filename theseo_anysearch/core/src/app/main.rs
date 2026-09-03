use std::time::Instant;

use eframe::egui;
use egui::{CentralPanel, PointerButton, Sense, SidePanel};
use glam::Vec2;

use crate::{
    app::ui::panels,
    camera::Camera,
    renderer::{self, RenderInput, RenderSettings, RenderStats},
    voxel::world::{Coord, WorldState, WORLD_CENTER},
};

pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Voxel Renderer",
        options,
        Box::new(|_cc| Ok(Box::new(VoxelApp::default()))),
    )
}

struct VoxelApp {
    world: WorldState,
    camera: Camera,
    render_settings: RenderSettings,
    selected: Option<Coord>,
    editor_xyz: [u16; 3],
    render_stats: RenderStats,
    fps: f32,
    last_frame_time: Instant,
}

impl Default for VoxelApp {
    fn default() -> Self {
        Self {
            world: WorldState::default(),
            camera: Camera::default(),
            render_settings: RenderSettings::default(),
            selected: None,
            editor_xyz: [WORLD_CENTER.0, WORLD_CENTER.1, WORLD_CENTER.2],
            render_stats: RenderStats::default(),
            fps: 0.0,
            last_frame_time: Instant::now(),
        }
    }
}

impl eframe::App for VoxelApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.last_frame_time)
            .as_secs_f32()
            .max(1.0 / 240.0);
        self.last_frame_time = now;
        self.fps = 1.0 / dt;

        self.handle_keyboard(ctx, dt);

        SidePanel::left("sidebar")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                let output = panels::draw_sidebar(
                    ui,
                    &self.world,
                    &mut self.camera,
                    &mut self.render_settings,
                    &mut self.selected,
                    &mut self.editor_xyz,
                    self.fps,
                    &self.render_stats,
                );

                if let Some((coord, filled)) = output.set_voxel {
                    self.world.set(coord, filled);
                }
            });

        CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
            let rect = response.rect;

            let scroll_delta = ctx.input(|i| i.smooth_scroll_delta.y);
            if response.hovered() && scroll_delta.abs() > 0.0 {
                self.camera.zoom(scroll_delta);
            }

            let pointer_delta = ctx.input(|i| i.pointer.delta());
            let primary_down = ctx.input(|i| i.pointer.primary_down());
            let secondary_down = ctx.input(|i| i.pointer.secondary_down());

            if response.hovered() && primary_down {
                self.camera
                    .orbit(Vec2::new(pointer_delta.x, pointer_delta.y));
            }
            if response.hovered() && secondary_down {
                self.camera.pan(Vec2::new(pointer_delta.x, pointer_delta.y));
            }

            let render_input = RenderInput {
                pointer_pos: response.interact_pointer_pos(),
                left_click: response.clicked_by(PointerButton::Primary),
                right_click: response.clicked_by(PointerButton::Secondary),
                selected: self.selected,
            };

            let render_output = renderer::render_world(
                &painter,
                rect,
                &self.world,
                &self.camera,
                &self.render_settings,
                &render_input,
            );

            if let Some(coord) = render_output.left_clicked_voxel {
                self.selected = Some(coord);
                self.editor_xyz = [coord.0, coord.1, coord.2];
            }

            if let Some(coord) = render_output.right_clicked_voxel {
                self.world.toggle(coord);
            }

            self.render_stats = render_output.stats;
        });

        ctx.request_repaint();
    }
}

impl VoxelApp {
    fn handle_keyboard(&mut self, ctx: &egui::Context, dt: f32) {
        let mut forward = 0.0;
        let mut strafe = 0.0;

        ctx.input(|i| {
            if i.key_pressed(egui::Key::R) {
                self.camera.reset();
            }
            if i.key_pressed(egui::Key::G) {
                self.render_settings.show_grid = !self.render_settings.show_grid;
            }
            if i.key_pressed(egui::Key::Escape) {
                self.selected = None;
            }
            if i.key_down(egui::Key::W) {
                forward += 1.0;
            }
            if i.key_down(egui::Key::S) {
                forward -= 1.0;
            }
            if i.key_down(egui::Key::A) {
                strafe -= 1.0;
            }
            if i.key_down(egui::Key::D) {
                strafe += 1.0;
            }
        });

        if forward != 0.0 || strafe != 0.0 {
            self.camera.move_local(forward, strafe, dt);
        }
    }
}
