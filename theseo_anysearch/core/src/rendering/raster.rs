use image::{Rgba, RgbaImage};

use super::projection::{is_face_visible, project_with_depth, CameraAngles};

const DEPTH_NEAR_IS_GREATER: bool = true;

pub(crate) fn draw_square(img: &mut RgbaImage, x: i32, y: i32, radius: i32, color: [u8; 4]) {
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
    let max_x =
        a.x.max(b.x)
            .max(c.x)
            .ceil()
            .min((img.width().saturating_sub(1)) as f32) as i32;
    let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as i32;
    let max_y =
        a.y.max(b.y)
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

pub(crate) fn draw_voxel_cube(
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
        Vertex {
            x: sx,
            y: sy,
            z: pz,
        }
    };

    // All 6 face definitions: (normal, [corner offsets], shade_multiplier, depth_bias)
    // Only camera-facing faces are drawn (back-face culling via is_face_visible).
    let faces: [((f32, f32, f32), [(f32, f32, f32); 4], f32, f32); 6] = [
        (
            (0.0, 1.0, 0.0),
            [(0., 1., 0.), (1., 1., 0.), (1., 1., 1.), (0., 1., 1.)],
            1.18,
            -0.02,
        ), // top (+Y)
        (
            (0.0, -1.0, 0.0),
            [(0., 0., 0.), (1., 0., 0.), (1., 0., 1.), (0., 0., 1.)],
            0.60,
            0.0,
        ), // bottom (-Y)
        (
            (1.0, 0.0, 0.0),
            [(1., 0., 0.), (1., 1., 0.), (1., 1., 1.), (1., 0., 1.)],
            0.92,
            0.0,
        ), // +X
        (
            (-1.0, 0.0, 0.0),
            [(0., 0., 0.), (0., 1., 0.), (0., 1., 1.), (0., 0., 1.)],
            0.70,
            0.0,
        ), // -X
        (
            (0.0, 0.0, 1.0),
            [(0., 0., 1.), (0., 1., 1.), (1., 1., 1.), (1., 0., 1.)],
            0.76,
            0.0,
        ), // +Z
        (
            (0.0, 0.0, -1.0),
            [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)],
            0.84,
            0.0,
        ), // -Z
    ];

    for (normal, offsets, shade, bias) in faces {
        if !disable_culling && !is_face_visible(normal, cam) {
            continue;
        }
        let p: [Vertex; 4] = offsets.map(|(ox, oy, oz)| mapv((x + ox, y + oy, z + oz)));
        draw_quad_depth(
            img,
            zbuf,
            p[0],
            p[1],
            p[2],
            p[3],
            blend(base_color, shade),
            DEPTH_NEAR_IS_GREATER,
            bias,
        );
    }
}
