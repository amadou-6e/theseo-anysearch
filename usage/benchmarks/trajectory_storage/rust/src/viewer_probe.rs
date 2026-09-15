// Reuse the actual viewer records, mutation resolver, mesher, depth sorting,
// and scene drawing. No production format migration or duplicated renderer.
#![allow(dead_code, unused_variables)]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../theseo_anysearch/core/src/bin/voxel_replay.rs"));

pub struct Trace(TrajectoryData);

pub fn parse(value: serde_json::Value) -> Result<Trace, serde_json::Error> {
    serde_json::from_value(value).map(Trace)
}

pub fn query(trace: &Trace, step: usize) -> usize {
    let record = std::hint::black_box(&trace.0.episode.steps[step]);
    std::hint::black_box(record.cursor_x as usize + record.cursors.len())
}

pub fn overlay(trace: &Trace, step: usize) -> usize {
    std::hint::black_box(replay_mutations_at(&trace.0.episode.steps, step)).len()
}

pub fn artifact_probe(path: &std::path::Path) -> Result<serde_json::Value, String> {
    let started = Instant::now();
    let trajectory = load_trajectory(path).ok_or("actual viewer rejected original artifact")?;
    let artifact_ms = started.elapsed().as_secs_f64() * 1000.0;
    let count = trajectory.episode.steps.len();
    let started = Instant::now();
    let mut app = VoxelReplayApp::new(vec![trajectory], None);
    let setup_ms = started.elapsed().as_secs_f64() * 1000.0;
    let source = app.regional_sources[0].clone().ok_or("compiled world unavailable")?;
    let ctx = egui::Context::default();
    let mut samples = Vec::new();
    for index in [0, count / 4, count / 2, count - 1] {
        let started = Instant::now();
        let center = selected_agent_center(&app.trajectories[0], index).ok_or("no agent center")?;
        let mutations = replay_mutations_at(&app.trajectories[0].episode.steps, index);
        let frame = source.load_agent_region(center, 16, &mutations).map_err(|e| format!("{e:?}"))?;
        app.cache_regional_faces(RegionRequestKey {
            iteration: 0, step: index, center, radius: 16, camera_revision: 0,
        }, &frame);
        let origin = agent_view_render_origin(center, 16);
        let mut items = app.regional_faces.iter().copied()
            .filter(|face| frame.region.contains(face.voxel))
            .map(SceneDrawItem::WorldFace).collect::<Vec<_>>();
        items.push(SceneDrawItem::Cursor(center));
        sort_scene_items(&mut items, origin, &app.camera);
        let load_mesh_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let output = ctx.run(egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1024.0, 768.0))),
            ..Default::default()
        }, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
                let rect = response.rect.shrink(20.0);
                let bounds = app.camera.bounds(33.0);
                draw_grid_bounds_layer(&painter, rect, &app.camera, &bounds, 33.0, false);
                for &item in &items {
                    item.draw(&painter, origin, rect, &app.camera, &bounds,
                              Color32::from_rgb(120, 120, 130), false);
                }
                draw_grid_bounds_layer(&painter, rect, &app.camera, &bounds, 33.0, true);
            });
        });
        let meshes = ctx.tessellate(output.shapes, output.pixels_per_point);
        std::hint::black_box(&meshes);
        samples.push(serde_json::json!({"step":index, "faces":app.regional_faces.len(),
            "region_overlay_mesh_ms":load_mesh_ms,
            "scene_paint_tessellate_ms":started.elapsed().as_secs_f64()*1000.0}));
    }
    Ok(serde_json::json!({"actual_artifact_load_ms":artifact_ms,
        "actual_viewer_world_setup_ms":setup_ms, "cpu_scene_samples":samples,
        "limits":"Actual viewer helpers, synchronous radius-16 regional load and CPU tessellation; no GPU upload/presentation, complete sidebar UI, or async LOD scheduling. Scene samples use the original verified artifact; scene work is format-independent after materialization."}))
}
