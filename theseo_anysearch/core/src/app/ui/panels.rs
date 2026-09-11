use egui::{Slider, Ui};

use crate::{
    camera::Camera,
    renderer::{RenderSettings, RenderStats},
    voxel::world::{Coord, WorldState, WORLD_SIZE},
};

#[derive(Default)]
pub struct SidebarOutput {
    pub set_voxel: Option<(Coord, bool)>,
}

pub fn draw_sidebar(
    ui: &mut Ui,
    world: &WorldState,
    camera: &mut Camera,
    settings: &mut RenderSettings,
    selected: &mut Option<Coord>,
    editor_xyz: &mut [u16; 3],
    fps: f32,
    stats: &RenderStats,
) -> SidebarOutput {
    let mut output = SidebarOutput::default();

    ui.heading("Voxel Space");
    ui.separator();

    ui.label("World info");
    ui.monospace(format!(
        "Dimensions: {} x {} x {}",
        WORLD_SIZE, WORLD_SIZE, WORLD_SIZE
    ));
    ui.monospace(format!("Filled voxels: {}", world.len()));
    ui.monospace(format!(
        "Sparse memory (est): {:.2} KB",
        world.estimated_sparse_bytes() as f32 / 1024.0
    ));
    ui.monospace(format!(
        "Dense world bytes: {:.2} MB",
        WorldState::dense_world_bytes() as f32 / (1024.0 * 1024.0)
    ));

    ui.separator();
    ui.label("Camera info");
    ui.monospace(format!(
        "Pos: {:.1}, {:.1}, {:.1}",
        camera.position.x, camera.position.y, camera.position.z
    ));
    ui.monospace(format!(
        "Target: {:.1}, {:.1}, {:.1}",
        camera.target.x, camera.target.y, camera.target.z
    ));
    ui.add(Slider::new(&mut camera.fov_y_deg, 25.0..=120.0).text("FOV"));

    ui.separator();
    ui.label("Voxel editor");
    ui.add(Slider::new(&mut editor_xyz[0], 0..=WORLD_SIZE - 1).text("X"));
    ui.add(Slider::new(&mut editor_xyz[1], 0..=WORLD_SIZE - 1).text("Y"));
    ui.add(Slider::new(&mut editor_xyz[2], 0..=WORLD_SIZE - 1).text("Z"));

    if ui.button("Set voxel").clicked() {
        output.set_voxel = Some(((editor_xyz[0], editor_xyz[1], editor_xyz[2]), true));
    }
    if ui.button("Clear voxel").clicked() {
        output.set_voxel = Some(((editor_xyz[0], editor_xyz[1], editor_xyz[2]), false));
    }

    if let Some((x, y, z)) = selected {
        ui.monospace(format!("Selected: ({x}, {y}, {z})"));
    } else {
        ui.monospace("Selected: none");
    }

    ui.separator();
    ui.label("Render settings");
    ui.checkbox(&mut settings.face_culling, "Face culling");
    ui.checkbox(&mut settings.wireframe, "Wireframe");
    ui.checkbox(&mut settings.show_grid, "Grid");
    ui.color_edit_button_srgba(&mut settings.voxel_color);

    ui.separator();
    ui.label("Performance");
    ui.monospace(format!("FPS: {:.1}", fps));
    ui.monospace(format!("Visible voxels: {}", stats.visible_voxels));
    ui.monospace(format!("Visible faces: {}", stats.visible_faces));
    ui.monospace(format!("Render time: {:.3} ms", stats.render_time_ms));

    if ui.button("Reset camera (R)").clicked() {
        camera.reset();
    }

    output
}
