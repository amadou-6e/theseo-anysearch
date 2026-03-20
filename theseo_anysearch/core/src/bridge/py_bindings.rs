use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::{
    bridge::dto::{EnvConfig, GeometrySubmission},
    environments::{Environment, VoxelAction, VoxelEnv, SurfaceAction, SurfaceEnv},
    world::{
        ingest::{parse_ascii_stl, voxelize_mesh},
        Block, BlockUpdate, Coord, World, WorldState,
    },
};
use image::{
    codecs::gif::{GifEncoder, Repeat},
    Delay, Frame, Rgba, RgbaImage,
};
use pyo3::prelude::*;
use serde::Serialize;

// Larger zr2 = closer to camera (camera sits in the high-zr2 direction).
const DEPTH_NEAR_IS_GREATER: bool = true;

#[derive(Clone, Copy)]
struct CameraAngles {
    cos_yaw: f32,
    sin_yaw: f32,
    cos_pitch: f32,
    sin_pitch: f32,
}

impl CameraAngles {
    fn new(yaw_deg: f32, pitch_deg: f32) -> Self {
        let yaw = yaw_deg.to_radians();
        let pitch = pitch_deg.to_radians();
        Self {
            cos_yaw: yaw.cos(),
            sin_yaw: yaw.sin(),
            cos_pitch: pitch.cos(),
            sin_pitch: pitch.sin(),
        }
    }
}

pub fn run_submission_poc(
    submission: &GeometrySubmission,
    env_config: &EnvConfig,
) -> Result<String, String> {
    let mesh = parse_ascii_stl(&submission.stl_ascii).map_err(|e| format!("{e:?}"))?;
    let placements = voxelize_mesh(&mesh, submission.origin, submission.scale);

    let mut world = WorldState::new();
    for placement in &placements {
        world
            .set_block(placement.coord, placement.block.clone())
            .map_err(|e| format!("{e:?}"))?;
    }

    let mut env = VoxelEnv::new(world, env_config.max_steps);
    let _ = env.reset(42);
    let first = env.step(VoxelAction::Noop);

    Ok(format!(
        "submission={} triangles={} placements={} first_reward={:.3} filled={}",
        submission.name,
        mesh.triangles.len(),
        placements.len(),
        first.reward,
        first.observation.filled
    ))
}

#[pyfunction]
pub fn py_submit_geometry(
    name: String,
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    max_steps: u32,
) -> PyResult<String> {
    let submission = GeometrySubmission {
        name,
        stl_ascii,
        origin: (origin_x, origin_y, origin_z),
        scale,
    };
    let env_config = EnvConfig {
        max_steps,
    };
    run_submission_poc(&submission, &env_config)
        .map_err(|msg| pyo3::exceptions::PyRuntimeError::new_err(msg))
}

#[pyfunction]
pub fn py_world_smoke_test() -> PyResult<String> {
    let mut world = WorldState::new();

    world
        .set_block((1, 2, 3), Block::default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

    world
        .update_block(
            (1, 2, 3),
            BlockUpdate {
                kind: Some(7),
                active: Some(false),
                reward_weight: Some(3.0),
            },
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

    world
        .remove_block((1, 2, 3))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

    Ok(format!("world_interface_poc_ok filled={}", world.len()))
}

#[pyfunction]
pub fn py_surface_env_from_stl(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
) -> PyResult<String> {
    let mesh = parse_ascii_stl(&stl_ascii)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
    let placements = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
    let filled = placements.iter().map(|p| p.coord).collect::<Vec<_>>();

    let mut env = SurfaceEnv::from_filled_surface(&filled, agent_count, max_steps);
    let mut obs = env.reset(seed);
    let mut total_reward = 0.0f32;
    let mut steps = 0u32;

    while steps < max_steps && obs.reached_agents < obs.total_agents {
        let sr = env.step(SurfaceAction::StepAll);
        total_reward += sr.reward;
        obs = sr.observation;
        steps += 1;
        if sr.done {
            break;
        }
    }

    Ok(format!(
        "triangles={} filled={} surface={} agents={} reached={}/{} steps={} reward={:.3}",
        mesh.triangles.len(),
        filled.len(),
        env.surface_count(),
        obs.total_agents,
        obs.reached_agents,
        obs.total_agents,
        steps,
        total_reward
    ))
}

#[derive(Clone, Serialize)]
struct EpisodeStep {
    step: u32,
    agents: Vec<crate::environments::AgentState>,
    reached_agents: usize,
    new_paints: Vec<PaintVoxel>,
}

#[derive(Clone, Serialize)]
struct PaintVoxel {
    coord: (u16, u16, u16),
    owner: usize,
}

#[derive(Clone, Serialize)]
struct EpisodeTrace {
    triangles: usize,
    filled: usize,
    surface: usize,
    total_agents: usize,
    max_steps: u32,
    steps_executed: u32,
    reached_agents: usize,
    total_reward: f32,
    geometry_coords: Vec<(u16, u16, u16)>,
    surface_coords: Vec<(u16, u16, u16)>,
    trace: Vec<EpisodeStep>,
}

#[derive(Serialize)]
struct EpochSummary {
    epoch: u32,
    reached: usize,
    total_agents: usize,
    steps: u32,
    reward: f32,
    painted_voxels: usize,
}

#[derive(Serialize)]
struct TrainingSummary {
    status: String,
    iterations: u32,
    video_every: u32,
    output_dir: String,
    summary_file: String,
}

fn run_episode_trace(
    stl_ascii: &str,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
) -> Result<EpisodeTrace, String> {
    let mesh = parse_ascii_stl(stl_ascii).map_err(|e| format!("{e:?}"))?;
    let placements = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
    let filled = placements.iter().map(|p| p.coord).collect::<Vec<_>>();

    let mut env = SurfaceEnv::from_filled_surface(&filled, agent_count, max_steps);
    let mut obs = env.reset(seed);
    let mut total_reward = 0.0f32;
    let mut steps = 0u32;
    let mut seen_painted = HashSet::new();
    let mut first_paints = Vec::new();
    for (coord, owner) in env.trail_owner() {
        seen_painted.insert(*coord);
        first_paints.push(PaintVoxel {
            coord: *coord,
            owner: *owner,
        });
    }
    let mut trace = vec![EpisodeStep {
        step: 0,
        agents: obs.agents.clone(),
        reached_agents: obs.reached_agents,
        new_paints: first_paints,
    }];

    while steps < max_steps && obs.reached_agents < obs.total_agents {
        let sr = env.step(SurfaceAction::StepAll);
        total_reward += sr.reward;
        steps += 1;
        obs = sr.observation;
        let mut new_paints = Vec::new();
        for (coord, owner) in env.trail_owner() {
            if seen_painted.insert(*coord) {
                new_paints.push(PaintVoxel {
                    coord: *coord,
                    owner: *owner,
                });
            }
        }
        trace.push(EpisodeStep {
            step: steps,
            agents: obs.agents.clone(),
            reached_agents: obs.reached_agents,
            new_paints,
        });
        if sr.done {
            break;
        }
    }

    Ok(EpisodeTrace {
        triangles: mesh.triangles.len(),
        filled: filled.len(),
        surface: env.surface_count(),
        total_agents: obs.total_agents,
        max_steps,
        steps_executed: steps,
        reached_agents: obs.reached_agents,
        total_reward,
        geometry_coords: filled,
        surface_coords: env.surface_coords(),
        trace,
    })
}

fn draw_square(img: &mut RgbaImage, x: i32, y: i32, radius: i32, color: [u8; 4]) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let x0 = (x - radius).max(0);
    let x1 = (x + radius).min(width - 1);
    let y0 = (y - radius).max(0);
    let y1 = (y + radius).min(height - 1);
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            img.put_pixel(xx as u32, yy as u32, Rgba(color));
        }
    }
}

fn blend(base: [u8; 4], factor: f32) -> [u8; 4] {
    let f = factor.clamp(0.0, 2.0);
    [
        (base[0] as f32 * f).clamp(0.0, 255.0) as u8,
        (base[1] as f32 * f).clamp(0.0, 255.0) as u8,
        (base[2] as f32 * f).clamp(0.0, 255.0) as u8,
        base[3],
    ]
}

fn edge(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
}

#[derive(Clone, Copy)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

fn draw_triangle_depth(
    img: &mut RgbaImage,
    zbuf: &mut [f32],
    a: Vertex,
    b: Vertex,
    c: Vertex,
    color: [u8; 4],
    near_is_greater: bool,
    depth_bias: f32,
) {
    let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as i32;
    let max_x = a
        .x
        .max(b.x)
        .max(c.x)
        .ceil()
        .min((img.width().saturating_sub(1)) as f32) as i32;
    let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as i32;
    let max_y = a
        .y
        .max(b.y)
        .max(c.y)
        .ceil()
        .min((img.height().saturating_sub(1)) as f32) as i32;

    let area = edge((a.x, a.y), (b.x, b.y), (c.x, c.y));
    if area.abs() < f32::EPSILON {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge((b.x, b.y), (c.x, c.y), p);
            let w1 = edge((c.x, c.y), (a.x, a.y), p);
            let w2 = edge((a.x, a.y), (b.x, b.y), p);
            let inside = if area > 0.0 {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if inside {
                let l0 = w0 / area;
                let l1 = w1 / area;
                let l2 = w2 / area;
                let depth = l0 * a.z + l1 * b.z + l2 * c.z + depth_bias;
                let idx = y as usize * img.width() as usize + x as usize;
                // DEPTH_NEAR_IS_GREATER=true: larger zr2 = closer to camera. Pass when new depth is larger.
                let pass = if near_is_greater {
                    depth > zbuf[idx]
                } else {
                    depth < zbuf[idx]
                };
                if pass {
                    zbuf[idx] = depth;
                    img.put_pixel(x as u32, y as u32, Rgba(color));
                }
            }
        }
    }
}

fn draw_quad_depth(
    img: &mut RgbaImage,
    zbuf: &mut [f32],
    p0: Vertex,
    p1: Vertex,
    p2: Vertex,
    p3: Vertex,
    color: [u8; 4],
    near_is_greater: bool,
    depth_bias: f32,
) {
    draw_triangle_depth(img, zbuf, p0, p1, p2, color, near_is_greater, depth_bias);
    draw_triangle_depth(img, zbuf, p0, p2, p3, color, near_is_greater, depth_bias);
}

fn map_coord(
    coord: (u16, u16, u16),
    min_px: f32,
    max_px: f32,
    min_py: f32,
    max_py: f32,
    width: i32,
    height: i32,
    margin: i32,
    cam: CameraAngles,
) -> (i32, i32) {
    let (px, py) = project(coord, cam);
    let span_x = (max_px - min_px).abs().max(1.0);
    let span_y = (max_py - min_py).abs().max(1.0);
    let x = margin as f32 + ((px - min_px) / span_x) * ((width - 2 * margin) as f32);
    let y = margin as f32 + ((py - min_py) / span_y) * ((height - 2 * margin) as f32);
    (x.round() as i32, y.round() as i32)
}

fn project(coord: (u16, u16, u16), cam: CameraAngles) -> (f32, f32) {
    let (sx, sy, _) = project_with_depth((coord.0 as f32, coord.1 as f32, coord.2 as f32), cam);
    (sx, sy)
}

fn project_with_depth(coord: (f32, f32, f32), cam: CameraAngles) -> (f32, f32, f32) {
    let x = coord.0;
    let y = coord.1;
    let z = coord.2;

    // Yaw rotation around Y axis, then pitch rotation around X axis.
    let xr = x * cam.cos_yaw - z * cam.sin_yaw;
    let zr = x * cam.sin_yaw + z * cam.cos_yaw;
    let yr = y * cam.cos_pitch - zr * cam.sin_pitch;
    let zr2 = y * cam.sin_pitch + zr * cam.cos_pitch;

    // zr2 is depth: larger = closer to camera (DEPTH_NEAR_IS_GREATER = true).
    (xr, yr - zr2 * 0.15, zr2)
}

/// Back-face culling: returns true if the face normal points toward the camera.
/// Camera sits at world offset (cos_pitch*cos_yaw, sin_pitch, cos_pitch*sin_yaw).
/// A face is front-facing when dot(normal, camera_offset_dir) > 0.
fn is_face_visible(normal: (f32, f32, f32), cam: CameraAngles) -> bool {
    let (nx, ny, nz) = normal;
    nx * cam.cos_pitch * cam.cos_yaw
        + ny * cam.sin_pitch
        + nz * cam.cos_pitch * cam.sin_yaw
        > 0.0
}

fn draw_voxel_cube(
    img: &mut RgbaImage,
    zbuf: &mut [f32],
    coord: (u16, u16, u16),
    base_color: [u8; 4],
    min_px: f32,
    max_px: f32,
    min_py: f32,
    max_py: f32,
    width: i32,
    height: i32,
    margin: i32,
    cam: CameraAngles,
    disable_culling: bool,
) {
    let x = coord.0 as f32;
    let y = coord.1 as f32;
    let z = coord.2 as f32;

    let mapv = |v: (f32, f32, f32)| -> Vertex {
        let (px, py, pz) = project_with_depth(v, cam);
        let span_x = (max_px - min_px).abs().max(1.0);
        let span_y = (max_py - min_py).abs().max(1.0);
        let sx = margin as f32 + ((px - min_px) / span_x) * ((width - 2 * margin) as f32);
        let sy = margin as f32 + ((py - min_py) / span_y) * ((height - 2 * margin) as f32);
        Vertex { x: sx, y: sy, z: pz }
    };

    // All 6 face definitions: (normal, [corner offsets], shade_multiplier, depth_bias)
    // Only camera-facing faces are drawn (back-face culling via is_face_visible).
    let faces: [((f32, f32, f32), [(f32, f32, f32); 4], f32, f32); 6] = [
        ((0.0, 1.0, 0.0),  [(0.,1.,0.),(1.,1.,0.),(1.,1.,1.),(0.,1.,1.)], 1.18, -0.02), // top (+Y)
        ((0.0, -1.0, 0.0), [(0.,0.,0.),(1.,0.,0.),(1.,0.,1.),(0.,0.,1.)], 0.60,  0.0),  // bottom (-Y)
        ((1.0, 0.0, 0.0),  [(1.,0.,0.),(1.,1.,0.),(1.,1.,1.),(1.,0.,1.)], 0.92,  0.0),  // +X
        ((-1.0, 0.0, 0.0), [(0.,0.,0.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.)], 0.70,  0.0),  // -X
        ((0.0, 0.0, 1.0),  [(0.,0.,1.),(0.,1.,1.),(1.,1.,1.),(1.,0.,1.)], 0.76,  0.0),  // +Z
        ((0.0, 0.0, -1.0), [(0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)], 0.84,  0.0),  // -Z
    ];

    for (normal, offsets, shade, bias) in faces {
        if !disable_culling && !is_face_visible(normal, cam) {
            continue;
        }
        let p: [Vertex; 4] = offsets.map(|(ox, oy, oz)| mapv((x + ox, y + oy, z + oz)));
        draw_quad_depth(img, zbuf, p[0], p[1], p[2], p[3], blend(base_color, shade), DEPTH_NEAR_IS_GREATER, bias);
    }
}

fn render_episode_gif(
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
        let mut img = RgbaImage::from_pixel(width as u32, height as u32, Rgba([248, 248, 248, 255]));
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
                &mut img, &mut zbuf, *coord, [140, 140, 150, 255],
                min_px, max_px, min_py, max_py, width, height, margin, cam, disable_culling,
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
            let (_, _, da) = project_with_depth((a.0 as f32 + 0.5, a.1 as f32 + 0.5, a.2 as f32 + 0.5), cam);
            let (_, _, db) = project_with_depth((b.0 as f32 + 0.5, b.1 as f32 + 0.5, b.2 as f32 + 0.5), cam);
            da.total_cmp(&db)
        });
        for (coord, owner) in paint_sorted {
            draw_voxel_cube(
                &mut img, &mut zbuf, coord, agent_colors[owner % agent_colors.len()],
                min_px, max_px, min_py, max_py, width, height, margin, cam, disable_culling,
            );
        }

        for (idx, agent) in step.agents.iter().enumerate() {
            let (tx, ty) =
                map_coord(agent.target, min_px, max_px, min_py, max_py, width, height, margin, cam);
            draw_square(&mut img, tx, ty, 3, [20, 20, 20, 255]);
            let (cx, cy) =
                map_coord(agent.current, min_px, max_px, min_py, max_py, width, height, margin, cam);
            draw_square(&mut img, cx, cy, 5, agent_colors[idx % agent_colors.len()]);
        }

        let frame = Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(83, 1));
        encoder
            .encode_frame(frame)
            .map_err(|e| format!("gif frame encode failed: {e}"))?;
    }

    Ok(())
}

#[pyfunction]
pub fn py_surface_env_trace(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
) -> PyResult<String> {
    let payload = run_episode_trace(
        &stl_ascii,
        origin_x,
        origin_y,
        origin_z,
        scale,
        agent_count,
        max_steps,
        seed,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    serde_json::to_string(&payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("serialize error: {e}")))
}

#[pyfunction]
pub fn py_train_videos(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
    iterations: u32,
    video_every: u32,
    output_dir: String,
    camera_yaw_deg: f32,
    camera_pitch_deg: f32,
    disable_culling: bool,
) -> PyResult<String> {
    let out_dir = Path::new(&output_dir);
    std::fs::create_dir_all(out_dir)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("mkdir failed: {e}")))?;

    let mut summary = Vec::new();
    for epoch in 1..=iterations {
        let trace = run_episode_trace(
            &stl_ascii,
            origin_x,
            origin_y,
            origin_z,
            scale,
            agent_count,
            max_steps,
            seed + epoch as u64,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let painted_voxels = trace
            .trace
            .iter()
            .map(|s| s.new_paints.len())
            .sum::<usize>();
        summary.push(EpochSummary {
            epoch,
            reached: trace.reached_agents,
            total_agents: trace.total_agents,
            steps: trace.steps_executed,
            reward: trace.total_reward,
            painted_voxels,
        });

        if video_every > 0 && epoch % video_every == 0 {
            let file = out_dir.join(format!("epoch_{epoch:04}.gif"));
            render_episode_gif(
                &trace,
                &file,
                camera_yaw_deg,
                camera_pitch_deg,
                1.1,
                disable_culling,
            )
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        }
    }

    let summary_file = out_dir.join("training_summary_rust.json");
    let payload = serde_json::to_string_pretty(&summary).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("summary serialize failed: {e}"))
    })?;
    std::fs::write(&summary_file, payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("summary write failed: {e}")))?;

    let response = TrainingSummary {
        status: "ok".to_string(),
        iterations,
        video_every,
        output_dir,
        summary_file: summary_file.to_string_lossy().to_string(),
    };
    serde_json::to_string(&response)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("response serialize failed: {e}")))
}

/// Render voxelized STL geometry (no agents) to a GIF for visual debugging.
#[pyfunction]
pub fn py_render_stl(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    output_path: String,
    camera_yaw_deg: f32,
    camera_pitch_deg: f32,
) -> PyResult<String> {
    let mesh = parse_ascii_stl(&stl_ascii)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
    let placements = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
    let geometry_coords: Vec<(u16, u16, u16)> = placements.iter().map(|p| p.coord).collect();
    let trace = EpisodeTrace {
        triangles: mesh.triangles.len(),
        filled: geometry_coords.len(),
        surface: 0,
        total_agents: 0,
        max_steps: 1,
        steps_executed: 1,
        reached_agents: 0,
        total_reward: 0.0,
        geometry_coords,
        surface_coords: vec![],
        trace: vec![EpisodeStep {
            step: 0,
            agents: vec![],
            reached_agents: 0,
            new_paints: vec![],
        }],
    };
    let path = std::path::Path::new(&output_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("mkdir failed: {e}")))?;
    }
    render_episode_gif(&trace, path, camera_yaw_deg, camera_pitch_deg, 1.1, false)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(format!(
        "stl {} triangles={} voxels={} yaw={} pitch={} -> {}",
        output_path, mesh.triangles.len(), trace.filled, camera_yaw_deg, camera_pitch_deg, output_path
    ))
}

// ---------------------------------------------------------------------------
// PyO3 class bindings — VoxelEnv
// ---------------------------------------------------------------------------

/// Python-visible observation returned by PyVoxelEnv.reset() and .step().
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyVoxelObservation {
    #[pyo3(get)]
    pub filled: usize,
    #[pyo3(get)]
    pub steps_remaining: u32,
    /// Manhattan distance to the active goal, or None if no goal is set.
    #[pyo3(get)]
    pub goal_distance: Option<u32>,
}

/// Python-visible step result returned by PyVoxelEnv.step().
#[pyclass]
pub struct PyStepResultVoxel {
    obs: PyVoxelObservation,
    #[pyo3(get)]
    pub reward: f32,
    #[pyo3(get)]
    pub done: bool,
}

#[pymethods]
impl PyStepResultVoxel {
    #[getter]
    fn observation(&self) -> PyVoxelObservation {
        self.obs.clone()
    }
}

/// Python-accessible wrapper around the Rust VoxelEnv.
///
/// Action space Discrete(26): all 26 neighbors in {-1,0,1}³ \ {origin}.
///
/// Movement: hitting the grid boundary → blocked (cursor stays, step consumed).
///           moving into a filled cell → blocked (cursor stays, step consumed, no penalty).
///           otherwise → cursor moves; trail_mode auto-fills destination.
#[pyclass]
pub struct PyVoxelEnv {
    inner: VoxelEnv,
}

#[pymethods]
impl PyVoxelEnv {
    /// * `max_steps`         – episode length limit.
    /// * `trail_mode`        – movement auto-fills the destination cell (default true).
    /// * `geometry`          – obstacle voxels pre-filled and restored on every reset.
    /// * `grid_size`         – side length of the cubic grid (default 32).
    /// * `step_cost`         – per-step penalty (default -0.01).
    /// * `goal_reward`       – bonus when cursor reaches goal (default 1.0).
    /// * `distance_shaping`  – L2 shaping coefficient toward goal (default 0.0).
    /// * `collision_cost`    – extra penalty subtracted on blocked moves (default 0.0).
    #[new]
    #[pyo3(signature = (max_steps, trail_mode=true, geometry=None,
                        grid_size=32, step_cost=-0.01, goal_reward=1.0,
                        distance_shaping=0.0, collision_cost=0.0))]
    pub fn new(
        max_steps: u32,
        trail_mode: bool,
        geometry: Option<Vec<(u16, u16, u16)>>,
        grid_size: u16,
        step_cost: f32,
        goal_reward: f32,
        distance_shaping: f32,
        collision_cost: f32,
    ) -> Self {
        let reward_config = crate::environments::voxel_env::RewardConfig {
            step_cost,
            goal_reward,
            distance_shaping,
            collision_cost,
        };
        let env = VoxelEnv::new(WorldState::new(), max_steps)
            .with_grid_size(grid_size)
            .with_geometry(geometry.unwrap_or_default())
            .with_trail_mode(trail_mode)
            .with_reward_config(reward_config);
        Self { inner: env }
    }

    /// Reset the environment and return the initial observation.
    /// Cursor is placed at the randomly-selected (or fixed) start position.
    pub fn reset(&mut self, seed: u64) -> PyVoxelObservation {
        let obs = self.inner.reset(seed);
        PyVoxelObservation {
            filled: obs.filled,
            steps_remaining: obs.steps_remaining,
            goal_distance: obs.goal_distance,
        }
    }

    /// Step the environment.
    /// Action space: Discrete(26) — all 26 neighbors in {-1,0,1}³ \ {origin},
    /// enumerated as (dx,dy,dz) with dz inner loop, skip (0,0,0) at index 13.
    pub fn step(&mut self, action: i32) -> PyStepResultVoxel {
        let g = self.inner.grid_size as i32;
        let clamp = |v: i32| v.clamp(1, g) as u16;
        let (x, y, z) = self.inner.cursor();

        // Map action 0..25 → (dx, dy, dz) offset into the 26-neighbor cube.
        // Enumerate all (dx,dy,dz) in {-1,0,1}³, outer→inner: dx, dy, dz; skip (0,0,0).
        let offsets: [(i32, i32, i32); 26] = {
            let mut arr = [(0i32, 0i32, 0i32); 26];
            let mut i = 0;
            let mut idx = 0i32;
            for dx in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dz in -1i32..=1 {
                        if dx == 0 && dy == 0 && dz == 0 { idx += 1; continue; }
                        arr[i] = (dx, dy, dz);
                        i += 1;
                        idx += 1;
                    }
                }
            }
            arr
        };

        let dest: (u16, u16, u16) = if (action as usize) < 26 {
            let (dx, dy, dz) = offsets[action as usize];
            (clamp(x as i32 + dx), clamp(y as i32 + dy), clamp(z as i32 + dz))
        } else {
            (x, y, z) // unknown action → treat as boundary collision
        };

        let rust_action = if dest == (x, y, z) {
            // Hit grid boundary or unknown action — blocked.
            VoxelAction::Collision
        } else if self.inner.world().is_filled(dest) {
            // Occupied cell — blocked, cursor stays.
            VoxelAction::Collision
        } else {
            // Successful move: update cursor first so step() sees new position.
            self.inner.set_cursor(dest);
            if self.inner.trail_mode() {
                VoxelAction::Place(dest)  // trail: auto-fill destination
            } else {
                VoxelAction::Noop
            }
        };

        let result = self.inner.step(rust_action);
        PyStepResultVoxel {
            obs: PyVoxelObservation {
                filled: result.observation.filled,
                steps_remaining: result.observation.steps_remaining,
                goal_distance: result.observation.goal_distance,
            },
            reward: result.reward,
            done: result.done,
        }
    }

    /// Returns the current cursor position as (x, y, z) in [1, 32].
    pub fn cursor_pos(&self) -> (u16, u16, u16) {
        self.inner.cursor()
    }

    /// Returns the active goal position for this episode, or None if unset.
    pub fn goal_pos(&self) -> Option<(u16, u16, u16)> {
        self.inner.active_goal()
    }

    /// Fix start and goal positions for all subsequent episodes.
    /// Overrides random surface-cell selection.
    pub fn set_waypoints(&mut self, start: (u16, u16, u16), goal: (u16, u16, u16)) {
        self.inner.set_waypoints(start, goal);
    }

    /// Clear fixed waypoints so random selection resumes.
    pub fn clear_waypoints(&mut self) {
        self.inner.clear_waypoints();
    }

    /// Returns a flattened (2*radius+1)³ binary array centred on the cursor.
    /// Each element is 1.0 if the world cell at (cursor + offset) is filled, else 0.0.
    /// Cells outside world bounds [0, 999] are 0.0.
    /// Elements are ordered x-outer, y-mid, z-inner (x changes slowest).
    pub fn box_obs(&self, radius: u32) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.cursor();
        let r = radius as i32;
        let side = (2 * r + 1) as usize;
        let mut result = Vec::with_capacity(side * side * side);
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    let x = cx as i32 + dx;
                    let y = cy as i32 + dy;
                    let z = cz as i32 + dz;
                    let filled = if x >= 0 && y >= 0 && z >= 0
                        && x < 1000 && y < 1000 && z < 1000
                    {
                        self.inner.world().is_filled((x as u16, y as u16, z as u16)) as u8 as f32
                    } else {
                        0.0
                    };
                    result.push(filled);
                }
            }
        }
        result
    }

    /// Returns a 27-element proximity array, one per direction
    /// (dx, dy, dz) ∈ {-1,0,+1}³ sorted lexicographically on (dx+1, dy+1, dz+1).
    ///
    /// Encoding:
    ///   0.0                              — no hit within max_len (or cursor cell empty for (0,0,0))
    ///   1.0 - (d - 1) as f32 / max_len  — hit at step d  (d ∈ 1..=max_len)
    ///   1.0                              — (0,0,0) direction: cursor cell is filled
    ///
    /// Index 13 = direction (0,0,0). Lower value = no detection; higher = closer hit.
    pub fn radial_obs(&self, max_len: u32) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.cursor();
        let mut result = Vec::with_capacity(27);
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        let self_filled = self.inner.world().is_filled((cx, cy, cz)) as u8 as f32;
                        result.push(self_filled);
                    } else {
                        let mut proximity = 0.0f32;
                        for step in 1..=max_len {
                            let nx = cx as i32 + dx * step as i32;
                            let ny = cy as i32 + dy * step as i32;
                            let nz = cz as i32 + dz * step as i32;
                            if nx < 0 || ny < 0 || nz < 0
                                || nx >= 1000 || ny >= 1000 || nz >= 1000
                            {
                                break;
                            }
                            if self.inner.world().is_filled((nx as u16, ny as u16, nz as u16)) {
                                proximity = 1.0 - (step - 1) as f32 / max_len as f32;
                                break;
                            }
                        }
                        result.push(proximity);
                    }
                }
            }
        }
        result
    }

    /// Returns all currently filled voxel coordinates as a list of (x, y, z) tuples.
    /// Used to snapshot the world state at episode start for trajectory recording.
    pub fn filled_voxels(&self) -> Vec<(u16, u16, u16)> {
        self.inner.world().iter_filled().copied().collect()
    }
}


// ---------------------------------------------------------------------------
// PyO3 class bindings — SurfaceEnv
// ---------------------------------------------------------------------------

/// Python-visible per-agent state inside a SurfaceObservation.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyAgentState {
    #[pyo3(get)]
    pub current: (u16, u16, u16),
    #[pyo3(get)]
    pub target: (u16, u16, u16),
    #[pyo3(get)]
    pub reached: bool,
}

// ---------------------------------------------------------------------------
// MultiVoxelEnv PyO3 types
// ---------------------------------------------------------------------------

/// Observation returned by PyMultiVoxelEnv.reset() and embedded in step results.
/// `cursors[i]` is agent i's (x, y, z) position.
/// `goal_distances[i]` is Some(manhattan distance) when a goal is active, else None.
#[pyclass]
#[derive(Clone)]
pub struct PyMultiVoxelObs {
    #[pyo3(get)]
    pub steps_remaining: u32,
    #[pyo3(get)]
    pub voxel_count: usize,
    #[pyo3(get)]
    pub cursors: Vec<(u16, u16, u16)>,
    #[pyo3(get)]
    pub goal_distances: Vec<Option<u32>>,
}

/// Step result returned by PyMultiVoxelEnv.step().
#[pyclass]
pub struct PyMultiVoxelStepResult {
    #[pyo3(get)]
    pub observation: PyMultiVoxelObs,
    #[pyo3(get)]
    pub rewards: Vec<f32>,
    #[pyo3(get)]
    pub done: bool,
}

/// Multi-agent Voxel environment: N agents share a configurable cubic grid.
/// Action space per agent: Discrete(26) — all 26 face/edge/corner neighbors.
#[pyclass]
pub struct PyMultiVoxelEnv {
    inner: crate::environments::multi_voxel_env::MultiAgentVoxelEnv,
}

#[pymethods]
impl PyMultiVoxelEnv {
    #[new]
    #[pyo3(signature = (agent_count, max_steps, trail_mode=true, geometry=None,
                        grid_size=32, step_cost=-0.01, goal_reward=1.0,
                        distance_shaping=0.0, collision_cost=0.0))]
    pub fn new(
        agent_count: usize,
        max_steps: u32,
        trail_mode: bool,
        geometry: Option<Vec<(u16, u16, u16)>>,
        grid_size: u16,
        step_cost: f32,
        goal_reward: f32,
        distance_shaping: f32,
        collision_cost: f32,
    ) -> Self {
        let reward_config = crate::environments::voxel_env::RewardConfig {
            step_cost, goal_reward, distance_shaping, collision_cost,
        };
        let inner = crate::environments::multi_voxel_env::MultiAgentVoxelEnv::new(
            agent_count,
            max_steps,
            trail_mode,
            geometry.unwrap_or_default(),
            reward_config,
            grid_size,
        );
        Self { inner }
    }

    pub fn reset(&mut self, seed: u64) -> PyMultiVoxelObs {
        let (steps_remaining, voxel_count, cursors, goal_distances) = self.inner.reset(seed);
        PyMultiVoxelObs { steps_remaining, voxel_count, cursors, goal_distances }
    }

    /// `actions` must be a list of length == agent_count; each value is 0..25.
    pub fn step(&mut self, actions: Vec<i32>) -> PyMultiVoxelStepResult {
        let r = self.inner.step_all(&actions);
        PyMultiVoxelStepResult {
            observation: PyMultiVoxelObs {
                steps_remaining: r.steps_remaining,
                voxel_count: r.voxel_count,
                cursors: r.cursors,
                goal_distances: r.goal_distances,
            },
            rewards: r.rewards,
            done: r.done,
        }
    }

    pub fn agent_count(&self) -> usize {
        self.inner.agent_count()
    }

    /// Returns all currently filled voxel coordinates (geometry + agent trail).
    pub fn filled_voxels(&self) -> Vec<(u16, u16, u16)> {
        self.inner.world.iter_filled().copied().collect()
    }

    /// Returns each agent's current cursor position.
    pub fn cursor_positions(&self) -> Vec<(u16, u16, u16)> {
        self.inner.agents.iter().map(|a| a.cursor).collect()
    }

    /// Returns each agent's current goal position, or None if unset.
    pub fn goal_positions(&self) -> Vec<Option<(u16, u16, u16)>> {
        self.inner.agents.iter().map(|a| a.goal).collect()
    }

    /// 6 binary values for agent `agent_idx`'s cardinal face-neighbors (+x,-x,+y,-y,+z,-z).
    /// 1.0 = that neighbor cell is filled, 0.0 = empty or out of bounds.
    pub fn face_neighbors(&self, agent_idx: usize) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.agents[agent_idx].cursor;
        let dirs: [(i32, i32, i32); 6] = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ];
        dirs.iter().map(|&(dx, dy, dz)| {
            let x = cx as i32 + dx;
            let y = cy as i32 + dy;
            let z = cz as i32 + dz;
            if x > 0 && y > 0 && z > 0 && x < 1000 && y < 1000 && z < 1000 {
                self.inner.world.is_filled((x as u16, y as u16, z as u16)) as u8 as f32
            } else {
                0.0
            }
        }).collect()
    }

    /// Flattened (2*radius+1)³ binary box observation centred on agent `agent_idx`'s cursor.
    /// 1.0 = filled, 0.0 = empty or out of bounds. Ordered x-outer, y-mid, z-inner.
    pub fn box_obs(&self, agent_idx: usize, radius: u32) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.agents[agent_idx].cursor;
        let r = radius as i32;
        let side = (2 * r + 1) as usize;
        let mut result = Vec::with_capacity(side * side * side);
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    let x = cx as i32 + dx;
                    let y = cy as i32 + dy;
                    let z = cz as i32 + dz;
                    let filled = if x > 0 && y > 0 && z > 0 && x < 1000 && y < 1000 && z < 1000 {
                        self.inner.world.is_filled((x as u16, y as u16, z as u16)) as u8 as f32
                    } else {
                        0.0
                    };
                    result.push(filled);
                }
            }
        }
        result
    }

    /// 27-element ray-cast observation from agent `agent_idx`'s cursor, one value per
    /// direction (dx,dy,dz) ∈ {-1,0,+1}³ sorted lexicographically on (dx+1,dy+1,dz+1).
    ///
    /// Encoding (distance, not proximity):
    ///   0.0                        — filled cell immediately adjacent (d=1) or self filled (0,0,0)
    ///   (d-1) as f32 / max_len     — hit at step d (d ∈ 2..=max_len); larger = farther
    ///   1.0                        — no hit within max_len, or self empty for (0,0,0)
    pub fn ray_cast(&self, agent_idx: usize, max_len: u32) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.agents[agent_idx].cursor;
        let mut result = Vec::with_capacity(27);
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        let filled = self.inner.world.is_filled((cx, cy, cz));
                        result.push(if filled { 0.0 } else { 1.0 });
                    } else {
                        let mut value = 1.0f32;
                        for step in 1..=max_len {
                            let nx = cx as i32 + dx * step as i32;
                            let ny = cy as i32 + dy * step as i32;
                            let nz = cz as i32 + dz * step as i32;
                            if nx <= 0 || ny <= 0 || nz <= 0 || nx >= 1000 || ny >= 1000 || nz >= 1000 {
                                break;
                            }
                            if self.inner.world.is_filled((nx as u16, ny as u16, nz as u16)) {
                                value = (step - 1) as f32 / max_len as f32;
                                break;
                            }
                        }
                        result.push(value);
                    }
                }
            }
        }
        result
    }
}

/// Python-visible observation returned by PySurfaceEnv.reset() and .step().
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySurfaceObservation {
    #[pyo3(get)]
    pub steps_remaining: u32,
    #[pyo3(get)]
    pub reached_agents: usize,
    #[pyo3(get)]
    pub total_agents: usize,
    #[pyo3(get)]
    pub agents: Vec<PyAgentState>,
}

/// Python-visible step result returned by PySurfaceEnv.step().
#[pyclass]
pub struct PyStepResultSurface {
    obs: PySurfaceObservation,
    #[pyo3(get)]
    pub reward: f32,
    #[pyo3(get)]
    pub done: bool,
}

#[pymethods]
impl PyStepResultSurface {
    #[getter]
    fn observation(&self) -> PySurfaceObservation {
        self.obs.clone()
    }
}

/// Python-accessible wrapper around the Rust SurfaceEnv.
///
/// All agents act simultaneously via StepAll — the `action` argument to
/// `step()` is ignored (only one joint action exists).
#[pyclass]
pub struct PySurfaceEnv {
    inner: SurfaceEnv,
}

#[pymethods]
impl PySurfaceEnv {
    /// Build from an explicit list of surface voxel coordinates.
    #[staticmethod]
    pub fn from_surface_coords(
        coords: Vec<(u16, u16, u16)>,
        agent_count: usize,
        max_steps: u32,
    ) -> Self {
        Self {
            inner: SurfaceEnv::from_surface(&coords, agent_count, max_steps),
        }
    }

    /// Build by voxelising an ASCII STL file and computing the exterior shell.
    #[staticmethod]
    pub fn from_stl(
        stl_ascii: &str,
        origin_x: u16,
        origin_y: u16,
        origin_z: u16,
        scale: f32,
        agent_count: usize,
        max_steps: u32,
    ) -> PyResult<Self> {
        let mesh = parse_ascii_stl(stl_ascii)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        let placements = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
        let filled: Vec<Coord> = placements.iter().map(|p| p.coord).collect();
        Ok(Self {
            inner: SurfaceEnv::from_filled_surface(&filled, agent_count, max_steps),
        })
    }

    /// Reset the environment and return the initial observation.
    pub fn reset(&mut self, seed: u64) -> PySurfaceObservation {
        let obs = self.inner.reset(seed);
        py_surface_obs(obs)
    }

    /// Step the environment (StepAll — `action` is ignored).
    pub fn step(&mut self, _action: i32) -> PyStepResultSurface {
        let result = self.inner.step(SurfaceAction::StepAll);
        PyStepResultSurface {
            obs: py_surface_obs(result.observation),
            reward: result.reward,
            done: result.done,
        }
    }
}

fn py_surface_obs(obs: crate::environments::SurfaceObservation) -> PySurfaceObservation {
    PySurfaceObservation {
        steps_remaining: obs.steps_remaining,
        reached_agents: obs.reached_agents,
        total_agents: obs.total_agents,
        agents: obs
            .agents
            .iter()
            .map(|a| PyAgentState {
                current: a.current,
                target: a.target,
                reached: a.reached,
            })
            .collect(),
    }
}

/// Render a standalone NxNxN solid cube to a GIF for visual debugging.
/// origin is the min-corner coordinate of the cube in world space.
#[pyfunction]
pub fn py_render_cube(
    output_path: String,
    cube_size: u16,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    camera_yaw_deg: f32,
    camera_pitch_deg: f32,
) -> PyResult<String> {
    let mut geometry_coords = Vec::new();
    for dz in 0..cube_size {
        for dy in 0..cube_size {
            for dx in 0..cube_size {
                geometry_coords.push((origin_x + dx, origin_y + dy, origin_z + dz));
            }
        }
    }
    let trace = EpisodeTrace {
        triangles: 0,
        filled: geometry_coords.len(),
        surface: 0,
        total_agents: 0,
        max_steps: 1,
        steps_executed: 1,
        reached_agents: 0,
        total_reward: 0.0,
        geometry_coords,
        surface_coords: vec![],
        trace: vec![EpisodeStep {
            step: 0,
            agents: vec![],
            reached_agents: 0,
            new_paints: vec![],
        }],
    };
    let path = std::path::Path::new(&output_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("mkdir failed: {e}")))?;
    }
    render_episode_gif(&trace, path, camera_yaw_deg, camera_pitch_deg, 3.0, false)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(format!(
        "cube {}^3 at ({},{},{}) yaw={} pitch={} -> {}",
        cube_size, origin_x, origin_y, origin_z, camera_yaw_deg, camera_pitch_deg, output_path
    ))
}

#[pymodule]
pub fn theseo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_submit_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(py_world_smoke_test, m)?)?;
    m.add_function(wrap_pyfunction!(py_surface_env_from_stl, m)?)?;
    m.add_function(wrap_pyfunction!(py_surface_env_trace, m)?)?;
    m.add_function(wrap_pyfunction!(py_train_videos, m)?)?;
    m.add_function(wrap_pyfunction!(py_render_stl, m)?)?;
    m.add_function(wrap_pyfunction!(py_render_cube, m)?)?;
    // VoxelEnv PyO3 classes
    m.add_class::<PyVoxelObservation>()?;
    m.add_class::<PyStepResultVoxel>()?;
    m.add_class::<PyVoxelEnv>()?;
    // MultiVoxelEnv PyO3 classes
    m.add_class::<PyMultiVoxelObs>()?;
    m.add_class::<PyMultiVoxelStepResult>()?;
    m.add_class::<PyMultiVoxelEnv>()?;
    // SurfaceEnv PyO3 classes
    m.add_class::<PyAgentState>()?;
    m.add_class::<PySurfaceObservation>()?;
    m.add_class::<PyStepResultSurface>()?;
    m.add_class::<PySurfaceEnv>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        environments::VoxelEnv,
        world::{Block, Coord, World, WorldState},
    };

    fn env_at_cursor(cursor: (u16, u16, u16)) -> PyVoxelEnv {
        let mut inner = VoxelEnv::new(WorldState::new(), 100);
        inner.set_cursor(cursor);
        PyVoxelEnv { inner }
    }

    fn env_with_block(cursor: (u16, u16, u16), coord: Coord) -> PyVoxelEnv {
        let mut world = WorldState::new();
        world.set_block(coord, Block::default()).unwrap();
        let mut inner = VoxelEnv::new(world, 100);
        inner.set_cursor(cursor);
        PyVoxelEnv { inner }
    }

    // ----- box_obs -----

    #[test]
    fn box_obs_empty_world_all_zeros() {
        let env = env_at_cursor((1, 1, 1)); // cursor at (1,1,1)
        let obs = env.box_obs(2);
        assert_eq!(obs.len(), 125);
        assert!(obs.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn box_obs_radius_0_single_cell() {
        let env = env_at_cursor((1, 1, 1)); // cursor at (1,1,1), empty world
        let obs = env.box_obs(0);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0], 0.0);
    }

    #[test]
    fn box_obs_single_filled_at_cursor() {
        // cursor at (1,1,1)
        let env = env_with_block((1, 1, 1), (1, 1, 1));
        let obs = env.box_obs(2); // 5³ = 125 cells
        // centre element: dx=0,dy=0,dz=0 → flat index = 2*25 + 2*5 + 2 = 62
        assert_eq!(obs[62], 1.0);
        let non_centre: Vec<f32> = obs.iter().enumerate()
            .filter(|&(i, _)| i != 62)
            .map(|(_, &v)| v)
            .collect();
        assert!(non_centre.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn box_obs_out_of_bounds_is_zero() {
        // cursor at (1,1,1); radius 2 → cells at coord 0 and -1 are OOB or edge
        // cell at (1-2, 1, 1) = (-1, 1, 1) — out of bounds (negative i32)
        let env = env_at_cursor((1, 1, 1)); // (1,1,1)
        let obs = env.box_obs(2);
        // first element: dx=-2,dy=-2,dz=-2 → (1-2,1-2,1-2) = (-1,-1,-1) → 0.0
        assert_eq!(obs[0], 0.0);
    }

    #[test]
    fn box_obs_length_correct_for_radius() {
        let env = env_at_cursor((1, 1, 1));
        assert_eq!(env.box_obs(0).len(), 1);
        assert_eq!(env.box_obs(1).len(), 27);
        assert_eq!(env.box_obs(2).len(), 125);
        assert_eq!(env.box_obs(3).len(), 343);
    }

    // ----- radial_obs -----

    #[test]
    fn radial_obs_length_is_27() {
        let env = env_at_cursor((1, 1, 1));
        assert_eq!(env.radial_obs(16).len(), 27);
    }

    #[test]
    fn radial_obs_empty_world_all_zeros() {
        let env = env_at_cursor((1, 1, 1));
        let obs = env.radial_obs(16);
        assert!(obs.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn radial_obs_self_cell_filled_index_13() {
        // cursor at (1,1,1), fill (1,1,1)
        let env = env_with_block((1, 1, 1), (1, 1, 1));
        let obs = env.radial_obs(16);
        assert_eq!(obs[13], 1.0); // index 13 = (0,0,0)
    }

    #[test]
    fn radial_obs_self_cell_empty_index_13() {
        let env = env_at_cursor((1, 1, 1)); // (1,1,1) empty
        let obs = env.radial_obs(16);
        assert_eq!(obs[13], 0.0);
    }

    #[test]
    fn radial_obs_adjacent_hit_proximity_is_1() {
        // cursor at (5,5,5), fill (6,5,5) — direction (+1,0,0), step d=1
        let env = env_with_block((5, 5, 5), (6, 5, 5));
        let obs = env.radial_obs(16);
        // direction (+1,0,0): index = (1+1)*9 + (0+1)*3 + (0+1) = 18+3+1 = 22
        assert_eq!(obs[22], 1.0);
    }

    #[test]
    fn radial_obs_distant_hit() {
        // cursor at (5,5,5), fill (13,5,5) — direction (+1,0,0), step d=8, max_len=16
        // expected: 1.0 - (8-1)/16.0 = 1.0 - 7/16.0 = 0.5625
        let env = env_with_block((5, 5, 5), (13, 5, 5));
        let obs = env.radial_obs(16);
        // direction (+1,0,0): index 22
        let expected = 1.0 - 7.0f32 / 16.0;
        assert!((obs[22] - expected).abs() < 1e-6);
    }

    #[test]
    fn radial_obs_hit_at_max_len_distinguishable() {
        // cursor at (5,5,5), fill (5+16,5,5) = (21,5,5) — direction (+1,0,0), d=16
        // expected: 1.0 - (16-1)/16.0 = 1/16.0 > 0
        let env = env_with_block((5, 5, 5), (21, 5, 5));
        let obs = env.radial_obs(16);
        let expected = 1.0f32 / 16.0;
        assert!((obs[22] - expected).abs() < 1e-6);
        assert!(obs[22] > 0.0);
    }

    #[test]
    fn radial_obs_no_hit_is_zero() {
        // empty world, direction (+1,0,0) should be 0.0
        let env = env_at_cursor((5, 5, 5));
        let obs = env.radial_obs(16);
        assert_eq!(obs[22], 0.0);
    }
}
