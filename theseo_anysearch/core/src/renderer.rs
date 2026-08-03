use std::time::Instant;

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke};
use glam::{Vec3, Vec4};

use crate::{
    camera::Camera,
    voxel::world::{Coord, WorldState, WORLD_CENTER},
};

#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub face_culling: bool,
    pub wireframe: bool,
    pub show_grid: bool,
    pub voxel_color: Color32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            face_culling: true,
            wireframe: false,
            show_grid: true,
            voxel_color: Color32::from_rgb(80, 170, 240),
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct RenderStats {
    pub visible_faces: usize,
    pub visible_voxels: usize,
    pub render_time_ms: f32,
}

pub struct RenderInput {
    pub pointer_pos: Option<Pos2>,
    pub left_click: bool,
    pub right_click: bool,
    pub selected: Option<Coord>,
}

#[derive(Default)]
pub struct RenderOutput {
    pub left_clicked_voxel: Option<Coord>,
    pub right_clicked_voxel: Option<Coord>,
    pub stats: RenderStats,
}

#[derive(Clone)]
struct FaceDraw {
    points: [Pos2; 4],
    // Average projected depth (NDC z). Larger values are farther for Mat4::perspective_rh.
    depth: f32,
    voxel: Coord,
    color: Color32,
}

const FACES: [([(i32, i32, i32); 4], (i32, i32, i32), f32); 6] = [
    ([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], (1, 0, 0), 1.0),
    (
        [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)],
        (-1, 0, 0),
        0.75,
    ),
    ([(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)], (0, 1, 0), 1.2),
    (
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (0, -1, 0),
        0.65,
    ),
    ([(1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)], (0, 0, 1), 0.9),
    (
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
        (0, 0, -1),
        0.8,
    ),
];

pub fn render_world(
    painter: &Painter,
    rect: Rect,
    world: &WorldState,
    camera: &Camera,
    settings: &RenderSettings,
    input: &RenderInput,
) -> RenderOutput {
    let start = Instant::now();
    let aspect = rect.width() / rect.height().max(1.0);
    let mvp = camera.projection_matrix(aspect) * camera.view_matrix();

    painter.rect_filled(rect, 0.0, Color32::from_rgb(12, 16, 22));

    if settings.show_grid {
        draw_grid(painter, rect, &mvp);
    }

    let mut faces = Vec::new();
    let mut visible_voxels = 0usize;

    for &(x, y, z) in world.iter_filled() {
        let center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
        if !in_view_frustum(center, &mvp) {
            continue;
        }
        visible_voxels += 1;

        for (corners, normal, shade) in FACES {
            if settings.face_culling {
                let nx = x as i32 + normal.0;
                let ny = y as i32 + normal.1;
                let nz = z as i32 + normal.2;
                if nx >= 0 && ny >= 0 && nz >= 0 {
                    let ncoord = (nx as u16, ny as u16, nz as u16);
                    if WorldState::in_bounds(ncoord) && world.is_filled(ncoord) {
                        continue;
                    }
                }
            }

            let mut projected = [Pos2::ZERO; 4];
            let mut projected_depth = [0.0_f32; 4];
            let mut failed = false;
            for (i, (ox, oy, oz)) in corners.into_iter().enumerate() {
                let p = Vec3::new(
                    x as f32 + ox as f32,
                    y as f32 + oy as f32,
                    z as f32 + oz as f32,
                );
                if let Some((screen, ndc_depth)) = project_to_screen_with_depth(p, rect, &mvp) {
                    projected[i] = screen;
                    projected_depth[i] = ndc_depth;
                } else {
                    failed = true;
                    break;
                }
            }
            if failed {
                continue;
            }

            // Use projected depth, not Euclidean distance, to avoid wrong overdraw at oblique angles.
            let depth = projected_depth.iter().copied().sum::<f32>() * 0.25;
            faces.push(FaceDraw {
                points: projected,
                depth,
                voxel: (x, y, z),
                color: shade_color(settings.voxel_color, shade),
            });
        }
    }

    // Far-to-near painter order by projected depth.
    faces.sort_by(|a, b| b.depth.total_cmp(&a.depth));

    for face in &faces {
        let stroke = if settings.wireframe {
            Stroke::new(1.0, Color32::BLACK)
        } else {
            Stroke::NONE
        };
        painter.add(Shape::convex_polygon(
            face.points.to_vec(),
            face.color,
            stroke,
        ));
    }

    if let Some(selected) = input.selected {
        draw_selected_outline(painter, rect, &mvp, selected);
    }

    let mut output = RenderOutput {
        left_clicked_voxel: None,
        right_clicked_voxel: None,
        stats: RenderStats {
            visible_faces: faces.len(),
            visible_voxels,
            render_time_ms: 0.0,
        },
    };

    if let Some(pointer) = input.pointer_pos {
        if let Some(hit) = pick_voxel(&faces, pointer) {
            if input.left_click {
                output.left_clicked_voxel = Some(hit);
            }
            if input.right_click {
                output.right_clicked_voxel = Some(hit);
            }
        }
    }

    output.stats.render_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    output
}

fn draw_grid(painter: &Painter, rect: Rect, mvp: &glam::Mat4) {
    let mut lines = Vec::new();
    let y = WORLD_CENTER.1 as f32;
    let min = 450;
    let max = 550;
    for x in (min..=max).step_by(5) {
        let p0 = Vec3::new(x as f32, y, min as f32);
        let p1 = Vec3::new(x as f32, y, max as f32);
        if let (Some(s0), Some(s1)) = (
            project_to_screen(p0, rect, mvp),
            project_to_screen(p1, rect, mvp),
        ) {
            lines.push((s0, s1));
        }
    }
    for z in (min..=max).step_by(5) {
        let p0 = Vec3::new(min as f32, y, z as f32);
        let p1 = Vec3::new(max as f32, y, z as f32);
        if let (Some(s0), Some(s1)) = (
            project_to_screen(p0, rect, mvp),
            project_to_screen(p1, rect, mvp),
        ) {
            lines.push((s0, s1));
        }
    }
    for (a, b) in lines {
        painter.line_segment([a, b], Stroke::new(1.0, Color32::from_gray(45)));
    }
}

fn draw_selected_outline(painter: &Painter, rect: Rect, mvp: &glam::Mat4, voxel: Coord) {
    let (x, y, z) = voxel;
    let corners = [
        Vec3::new(x as f32, y as f32, z as f32),
        Vec3::new(x as f32 + 1.0, y as f32, z as f32),
        Vec3::new(x as f32 + 1.0, y as f32 + 1.0, z as f32),
        Vec3::new(x as f32, y as f32 + 1.0, z as f32),
        Vec3::new(x as f32, y as f32, z as f32 + 1.0),
        Vec3::new(x as f32 + 1.0, y as f32, z as f32 + 1.0),
        Vec3::new(x as f32 + 1.0, y as f32 + 1.0, z as f32 + 1.0),
        Vec3::new(x as f32, y as f32 + 1.0, z as f32 + 1.0),
    ];

    let edges: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    for (a, b) in edges {
        let p0 = project_to_screen(corners[a], rect, mvp);
        let p1 = project_to_screen(corners[b], rect, mvp);
        if let (Some(s0), Some(s1)) = (p0, p1) {
            painter.line_segment([s0, s1], Stroke::new(2.0, Color32::YELLOW));
        }
    }
}

fn pick_voxel(faces: &[FaceDraw], pointer: Pos2) -> Option<Coord> {
    let mut best: Option<(Coord, f32)> = None;
    for face in faces {
        if point_in_quad(pointer, &face.points) {
            match best {
                Some((_, depth)) if face.depth >= depth => {}
                _ => best = Some((face.voxel, face.depth)),
            }
        }
    }
    best.map(|(coord, _)| coord)
}

fn in_view_frustum(center: Vec3, mvp: &glam::Mat4) -> bool {
    let clip = *mvp * Vec4::new(center.x, center.y, center.z, 1.0);
    if clip.w <= 0.001 {
        return false;
    }
    let ndc = clip.truncate() / clip.w;
    ndc.x.abs() <= 1.2 && ndc.y.abs() <= 1.2 && ndc.z >= -1.2 && ndc.z <= 1.2
}

fn project_to_screen(world: Vec3, rect: Rect, mvp: &glam::Mat4) -> Option<Pos2> {
    project_to_screen_with_depth(world, rect, mvp).map(|v| v.0)
}

fn project_to_screen_with_depth(world: Vec3, rect: Rect, mvp: &glam::Mat4) -> Option<(Pos2, f32)> {
    let clip = *mvp * Vec4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 0.001 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    // Mat4::perspective_rh uses depth in [0, 1], but we keep the lower bound tolerant for compatibility.
    if ndc.z < -1.0 || ndc.z > 1.0 {
        return None;
    }
    let sx = rect.left() + (ndc.x + 1.0) * 0.5 * rect.width();
    let sy = rect.top() + (1.0 - (ndc.y + 1.0) * 0.5) * rect.height();
    Some((Pos2::new(sx, sy), ndc.z))
}

fn shade_color(color: Color32, intensity: f32) -> Color32 {
    let rgba = color.to_array();
    let scale = intensity.clamp(0.0, 1.4);
    Color32::from_rgba_unmultiplied(
        ((rgba[0] as f32 * scale).clamp(0.0, 255.0)) as u8,
        ((rgba[1] as f32 * scale).clamp(0.0, 255.0)) as u8,
        ((rgba[2] as f32 * scale).clamp(0.0, 255.0)) as u8,
        rgba[3],
    )
}

fn point_in_quad(p: Pos2, quad: &[Pos2; 4]) -> bool {
    point_in_tri(p, quad[0], quad[1], quad[2]) || point_in_tri(p, quad[0], quad[2], quad[3])
}

fn point_in_tri(p: Pos2, a: Pos2, b: Pos2, c: Pos2) -> bool {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < f32::EPSILON {
        return false;
    }

    let inv = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;
    u >= 0.0 && v >= 0.0 && (u + v) <= 1.0
}
