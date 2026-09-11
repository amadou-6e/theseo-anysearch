use std::{collections::HashMap, fs::File, path::Path};

use image::{
    codecs::gif::{GifEncoder, Repeat},
    Delay, Frame, Rgba, RgbaImage,
};

use super::episode::EpisodeTrace;
use crate::rendering::{
    projection::{map_coord, project, project_with_depth, CameraAngles},
    raster::{draw_square, draw_voxel_cube},
};

const DEPTH_NEAR_IS_GREATER: bool = true;

pub(crate) fn render_episode_gif(
    trace: &EpisodeTrace,
    output_file: &Path,
    yaw_deg: f32,
    pitch_deg: f32,
    zoom_out: f32,
    disable_culling: bool,
) -> Result<(), String> {
    let cam = CameraAngles::new(yaw_deg, pitch_deg);
    let width = 720i32;
    let height = 720i32;
    let margin = 40i32;
    let agent_colors: [[u8; 4]; 5] = [
        [230, 25, 75, 255],
        [60, 180, 75, 255],
        [0, 130, 200, 255],
        [245, 130, 48, 255],
        [145, 30, 180, 255],
    ];

    let mut all_coords = trace.geometry_coords.clone();
    for step in &trace.trace {
        for a in &step.agents {
            all_coords.push(a.current);
            all_coords.push(a.target);
        }
    }
    if all_coords.is_empty() {
        return Err("no coordinates available for rendering".to_string());
    }

    let mut min_px = f32::INFINITY;
    let mut max_px = f32::NEG_INFINITY;
    let mut min_py = f32::INFINITY;
    let mut max_py = f32::NEG_INFINITY;
    for coord in &all_coords {
        let (px, py) = project(*coord, cam);
        min_px = min_px.min(px);
        max_px = max_px.max(px);
        min_py = min_py.min(py);
        max_py = max_py.max(py);
    }

    // Apply zoom_out: expand the bounding box symmetrically around its center.
    // Use the same half-span for X and Y to preserve aspect ratio (no stretching).
    let cx = (min_px + max_px) * 0.5;
    let cy = (min_py + max_py) * 0.5;
    let hx = (max_px - min_px).abs().max(1.0) * 0.5;
    let hy = (max_py - min_py).abs().max(1.0) * 0.5;
    let h = hx.max(hy) * zoom_out;
    let min_px = cx - h;
    let max_px = cx + h;
    let min_py = cy - h;
    let max_py = cy + h;

    let file = File::create(output_file).map_err(|e| format!("create gif failed: {e}"))?;
    let mut encoder = GifEncoder::new(file);
    encoder
        .set_repeat(Repeat::Infinite)
        .map_err(|e| format!("gif repeat failed: {e}"))?;

    let mut painted_owner: HashMap<(u16, u16, u16), usize> = HashMap::new();

    for step in &trace.trace {
        let mut img =
            RgbaImage::from_pixel(width as u32, height as u32, Rgba([248, 248, 248, 255]));
        let mut zbuf = if DEPTH_NEAR_IS_GREATER {
            vec![f32::NEG_INFINITY; (width * height) as usize]
        } else {
            vec![f32::INFINITY; (width * height) as usize]
        };

        let mut geom_sorted = trace
            .geometry_coords
            .iter()
            .map(|c| {
                let (_, _, depth) =
                    project_with_depth((c.0 as f32 + 0.5, c.1 as f32 + 0.5, c.2 as f32 + 0.5), cam);
                (*c, depth)
            })
            .collect::<Vec<_>>();
        // Sort far-to-near: ascending zr2 = far first (larger zr2 = closer). Z-buffer closes wins.
        geom_sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (coord, _) in &geom_sorted {
            draw_voxel_cube(
                &mut img,
                &mut zbuf,
                *coord,
                [140, 140, 150, 255],
                min_px,
                max_px,
                min_py,
                max_py,
                width,
                height,
                margin,
                cam,
                disable_culling,
            );
        }

        for paint in &step.new_paints {
            painted_owner.insert(paint.coord, paint.owner);
        }

        let mut paint_sorted = painted_owner
            .iter()
            .map(|(c, o)| (*c, *o))
            .collect::<Vec<_>>();
        paint_sorted.sort_by(|(a, _), (b, _)| {
            let (_, _, da) =
                project_with_depth((a.0 as f32 + 0.5, a.1 as f32 + 0.5, a.2 as f32 + 0.5), cam);
            let (_, _, db) =
                project_with_depth((b.0 as f32 + 0.5, b.1 as f32 + 0.5, b.2 as f32 + 0.5), cam);
            da.total_cmp(&db)
        });
        for (coord, owner) in paint_sorted {
            draw_voxel_cube(
                &mut img,
                &mut zbuf,
                coord,
                agent_colors[owner % agent_colors.len()],
                min_px,
                max_px,
                min_py,
                max_py,
                width,
                height,
                margin,
                cam,
                disable_culling,
            );
        }

        for (idx, agent) in step.agents.iter().enumerate() {
            let (tx, ty) = map_coord(
                agent.target,
                min_px,
                max_px,
                min_py,
                max_py,
                width,
                height,
                margin,
                cam,
            );
            draw_square(&mut img, tx, ty, 3, [20, 20, 20, 255]);
            let (cx, cy) = map_coord(
                agent.current,
                min_px,
                max_px,
                min_py,
                max_py,
                width,
                height,
                margin,
                cam,
            );
            draw_square(&mut img, cx, cy, 5, agent_colors[idx % agent_colors.len()]);
        }

        let frame = Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(83, 1));
        encoder
            .encode_frame(frame)
            .map_err(|e| format!("gif frame encode failed: {e}"))?;
    }

    Ok(())
}
